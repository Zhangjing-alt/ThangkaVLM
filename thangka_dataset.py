#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class ThangkaDataset(Dataset):
    """
    支持两种数据目录结构：
    Layout A（推荐/你的 split 后 test/val/train 也是这个）:
        data_dir/
          images/xxx.jpg
          annotations/xxx.json

    Layout B（平铺）:
        data_dir/
          xxx.jpg
          xxx.json
    """

    def __init__(
        self,
        data_dir: str,
        processor,
        tokenizer,
        split: str = "train",
        task_mode: str = "qa",  # "qa" | "detection" | "mixed"
        use_bbox: bool = True,
        augment_qa: bool = True,
        # 文本裁剪（只裁剪自然语言）
        user_max_tokens: int = 256,
        assistant_max_tokens: int = 256,
        # processor 之后截整段序列长度（强烈建议：1024/1536/2048）
        max_seq_len: int = 1024,
        # 强制统一图片尺寸（减少 vision tokens / image_pad tokens）
        image_max_side: int = 448,
        # 未 split 时内部按 8/1/1 切分；当 data_dir 名是 train/val/test 时自动禁用
        internal_split_ratio: tuple = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.tokenizer = tokenizer

        self.task_mode = task_mode
        self.use_bbox = use_bbox
        self.augment_qa = augment_qa

        self.user_max_tokens = user_max_tokens
        self.assistant_max_tokens = assistant_max_tokens

        self.max_seq_len = max_seq_len
        self.image_max_side = image_max_side

        self.internal_split_ratio = internal_split_ratio
        self.seed = seed

        if not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError("processor 没有 apply_chat_template（Qwen3-VL 的 processor 应该有这个方法）。")

        self.samples = self._load_dataset(split)
        print(f"✅ 加载了 {len(self.samples)} 条 {split} 数据 (data_dir={self.data_dir})")

    # -------------------------
    # 数据加载：兼容两种布局
    # -------------------------
    def _detect_layout(self) -> str:
        """返回 'A' 或 'B'"""
        anno_dir = self.data_dir / "annotations"
        img_dir = self.data_dir / "images"
        if anno_dir.exists() and img_dir.exists() and any(anno_dir.glob("*.json")):
            return "A"
        return "B"

    def _load_dataset(self, split: str) -> List[Dict]:
        layout = self._detect_layout()

        if layout == "A":
            image_dir = self.data_dir / "images"
            anno_dir = self.data_dir / "annotations"
            anno_files = sorted(anno_dir.glob("*.json"))
        else:
            # Layout B: 平铺目录
            image_dir = self.data_dir
            anno_files = sorted(self.data_dir.glob("*.json"))

        if not anno_files:
            return []

        # 如果 data_dir 的目录名已经是 train/val/test，则不再内部切分
        dir_name = self.data_dir.name.lower()
        already_split = dir_name in ["train", "val", "test"]

        rng = random.Random(self.seed)
        rng.shuffle(anno_files)

        if not already_split:
            total = len(anno_files)
            r0, r1, r2 = self.internal_split_ratio
            assert abs((r0 + r1 + r2) - 1.0) < 1e-6, "internal_split_ratio 必须加起来等于 1"

            i0 = int(r0 * total)
            i1 = int((r0 + r1) * total)

            if split == "train":
                anno_files = anno_files[:i0]
            elif split == "val":
                anno_files = anno_files[i0:i1]
            else:
                anno_files = anno_files[i1:]

        samples: List[Dict] = []
        for anno_file in anno_files:
            try:
                with open(anno_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            instances = data.get("instances", [])
            if not instances:
                continue

            for inst in instances:
                image_id = inst.get("image_id") or inst.get("image") or inst.get("filename")
                if not image_id:
                    continue

                image_path = image_dir / image_id
                if not image_path.exists():
                    # 兜底：如果 json 写的是不带后缀的 id
                    if "." not in image_id:
                        cand = image_dir / (image_id + ".jpg")
                        if cand.exists():
                            image_path = cand
                        else:
                            cand = image_dir / (image_id + ".png")
                            if cand.exists():
                                image_path = cand
                            else:
                                continue
                    else:
                        continue

                samples.append(
                    {
                        "image_path": str(image_path),
                        "image_id": image_path.name,
                        "elements": inst.get("elements", []),
                        "qa_pairs": inst.get("qa_pairs", []),
                    }
                )

        return samples

    def __len__(self):
        return len(self.samples)

    # -------------------------
    # 文本与样本构造
    # -------------------------
    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        if not text:
            return text
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        original_image_size = image.size   # (W,H) —— 必须在 resize 之前记录

        # 强制统一尺寸（减少视觉 token、避免多卡 batch 拼接差异）
        if self.image_max_side and self.image_max_side > 0:
            image = image.resize(
                (self.image_max_side, self.image_max_side),
                resample=Image.BICUBIC,
            )

        if self.task_mode == "qa":
            conversations = self._generate_qa_conversations(sample)
        elif self.task_mode == "detection":
            conversations = self._generate_detection_conversations(sample)
        else:
            conversations = self._generate_mixed_conversations(sample)

        if not conversations:
            conversations = [
                [
                    {"role": "user", "content": "请描述图片内容。"},
                    {"role": "assistant", "content": "这是一幅唐卡图像。"},
                ]
            ]

        conversation = random.choice(conversations)

        # messages（首轮 user 含 image + text）
        messages = []
        first_user = True
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                content = self._truncate_by_tokens(content, self.user_max_tokens)
            else:
                content = self._truncate_by_tokens(content, self.assistant_max_tokens)

            if role == "user":
                if first_user:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": content},
                            ],
                        }
                    )
                    first_user = False
                else:
                    messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})

        # 完整对话（user+assistant），用于训练输入
        prompt_full = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # user-only，用于 label mask（只监督 assistant）
        messages_user_only = []
        for m in messages:
            if m["role"] == "assistant":
                break
            messages_user_only.append(m)

        prompt_user_only = self.processor.apply_chat_template(
            messages_user_only, tokenize=False, add_generation_prompt=True
        )

        # 统一长度（避免多卡 batch 拼接问题）
        inputs = self.processor(
            images=image,
            text=prompt_full,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )

        # processor 输出是 batch=1，这里 squeeze 掉 batch 维
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # labels masking：只监督 assistant 输出
        labels = inputs["input_ids"].clone()
        user_only_ids = self.tokenizer(prompt_user_only, add_special_tokens=False)["input_ids"]
        cut = min(len(user_only_ids), labels.shape[0])
        labels[:cut] = -100
        inputs["labels"] = labels

        # ==========================================
        # 【修复版】支持像素坐标的 bbox（非归一化）
        # ==========================================
        # 默认占位
        # ===== 返修修复：region_labels 对齐到 merger 输出网格 =====
        # 关键：bbox 为原图像素坐标，必须用 original_image_size 归一化；
        #       token 数 N = t*(h//ms)*(w//ms)，而非 pixel_values.shape[0]
        from region_labels_utils import (normalize_boxes,
                                         build_region_labels_on_merged_grid)

        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is None:
            raise RuntimeError("processor 未返回 image_grid_thw，无法对齐 region_labels")
        merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        _t, _h, _w = [int(v) for v in torch.as_tensor(grid_thw).reshape(-1)[:3]]
        n_tokens = _t * (_h // merge_size) * (_w // merge_size)

        boxes = []
        if self.use_bbox and sample.get("elements"):
            boxes = normalize_boxes(sample["elements"], original_image_size)

        if boxes:
            inputs["region_labels"] = build_region_labels_on_merged_grid(
                boxes, grid_thw, merge_size)
        else:
            inputs["region_labels"] = torch.zeros(n_tokens, dtype=torch.long)

        assert inputs["region_labels"].numel() == n_tokens, (
            f"region_labels 长度 {inputs['region_labels'].numel()} != {n_tokens}")

        return inputs

    # -------------------------
    # 生成对话
    # -------------------------
    def _generate_qa_conversations(self, sample: Dict) -> List[List[Dict]]:
        conversations: List[List[Dict]] = []
        for qa in sample.get("qa_pairs", []):
            q = (qa.get("question") or "").strip()
            a = (qa.get("answer") or "").strip()
            if not q or not a:
                continue
            conversations.append(
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            )
        if self.augment_qa:
            conversations.extend(self._augment_qa_pairs(sample))
        return conversations

    def _augment_qa_pairs(self, sample: Dict) -> List[List[Dict]]:
        # 轻量增强：把 elements 中的 name_zh 做 yes/no 问句
        augmented: List[List[Dict]] = []
        elements = sample.get("elements", [])
        for e in elements:
            name = (e.get("name_zh") or e.get("label") or "").strip()
            if not name:
                continue
            augmented.append(
                [
                    {"role": "user", "content": f"图中是否有{name}？"},
                    {"role": "assistant", "content": "是"},
                ]
            )
        return augmented

    def _generate_detection_conversations(self, sample: Dict) -> List[List[Dict]]:
        conversations: List[List[Dict]] = []
        elements = sample.get("elements", [])
        if not elements:
            return conversations
        names = []
        for e in elements:
            name = (e.get("name_zh") or e.get("label") or "").strip()
            if name:
                names.append(name)
        if names:
            conversations.append(
                [
                    {"role": "user", "content": "图中包含哪些元素？"},
                    {"role": "assistant", "content": f"图中包含: {', '.join(names)}"},
                ]
            )
        return conversations

    def _generate_mixed_conversations(self, sample: Dict) -> List[List[Dict]]:
        conversations: List[List[Dict]] = []
        conversations.extend(self._generate_qa_conversations(sample))
        conversations.extend(self._generate_detection_conversations(sample))
        return conversations
