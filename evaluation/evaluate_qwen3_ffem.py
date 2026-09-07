#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# Register Qwen3VLWithFFEM so AutoModel uses the FFEM-enabled model class.
from model_registration import Qwen3VLWithFFEM
from modeling_qwen3_vl_ffem import generate_region_labels_normalized


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--ffem_path", required=True,
                   help="Path to pytorch_model_pure_ffem.bin")
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--image_max_side", type=int, default=448)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = full test set; positive integer = smoke test QA count")
    return p.parse_args()


def normalize_answer(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text)).strip().lower()

    for prefix in ["答案是", "答案为", "答：", "回答：", "the answer is", "answer:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    yes_set = {"是", "是的", "有", "存在", "正确", "yes", "true"}
    no_set = {"否", "不是", "没有", "不存在", "错误", "no", "false"}

    if text in yes_set:
        return "__yes__"
    if text in no_set:
        return "__no__"

    text = re.sub(
        r"""[\s\.,，。!！?？:：;；'"“”‘’、\-_/\\()\[\]{}<>《》]+""",
        "",
        text
    )
    return text



def is_correct_answer(pred: str, gt: str) -> bool:
    """
    Unified strict Thangka-VQA correctness protocol.

    - Professional names are strict: no synonym expansion.
    - Longer explanatory responses are allowed if the canonical GT answer
      is explicitly present and not negated.
    - Yes/No questions are judged by response polarity.
    - Accuracy evaluates the requested answer slot only; hallucination
      from additional unsupported claims is evaluated separately.
    """
    p = normalize_answer(pred)
    g = normalize_answer(gt)

    if not p or not g:
        return False

    raw = str(pred).strip().lower()

    # ============================================================
    # 1. Yes / No polarity
    # ============================================================
    if g == "__yes__":
        negatives = [
            "不是", "并非", "并不是",
            "没有", "并没有",
            "不存在", "不含", "不包含",
            "未见", "没有发现",
        ]

        if any(x in raw for x in negatives):
            return False

        positive_starts = (
            "是", "是的",
            "有", "有的",
            "存在",
            "确实", "的确",
            "yes", "true",
        )
        return raw.startswith(positive_starts)

    if g == "__no__":
        negative_starts = (
            "否",
            "没有", "不存在", "无",
            "不是", "并非", "并不是",
            "不含", "不包含", "未见",
            "no", "false",
        )
        return raw.startswith(negative_starts)

    # ============================================================
    # 2. Professional entity / attribute answer
    # ============================================================
    if p == g:
        return True

    # Allow the exact canonical GT term inside a longer explanation.
    # No synonym substitution is performed.
    if len(g) >= 2 and g in p:
        negated_patterns = [
            "不是" + g,
            "并非" + g,
            "并不是" + g,
            "没有" + g,
            "不存在" + g,
            "不含" + g,
            "不包含" + g,
        ]

        if any(x in p for x in negated_patterns):
            return False

        return True

    return False

def resize_for_eval(img, image_max_side):
    """
    Match training preprocessing exactly:
    force image to image_max_side x image_max_side using BICUBIC.
    """
    return img.resize(
        (image_max_side, image_max_side),
        resample=Image.BICUBIC,
    )


def load_test_samples(test_dir):
    root = Path(test_dir)
    ann_dir = root / "annotations"
    img_dir = root / "images"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Missing annotation dir: {ann_dir}")

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing image dir: {img_dir}")

    samples = []

    for ann_path in sorted(ann_dir.glob("*.json")):
        obj = json.loads(ann_path.read_text(encoding="utf-8"))

        instances = obj.get("instances", [])
        if not instances:
            continue

        for inst_idx, inst in enumerate(instances):
            image_id = inst.get("image_id")
            if not image_id:
                continue

            image_path = img_dir / image_id
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found for {ann_path.name}: {image_path}"
                )

            qa_pairs = inst.get("qa_pairs", [])

            for qa_idx, qa in enumerate(qa_pairs):
                question = qa.get("question")
                answer = qa.get("answer")

                if question is None or answer is None:
                    continue

                samples.append({
                    "id": f"{ann_path.stem}_{inst_idx}_{qa_idx}",
                    "annotation_file": ann_path.name,
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "question": question,
                    "ground_truth": answer,
                    "question_type": qa.get("question_type", "unknown"),
                    "is_mirrored": inst.get("is_mirrored", False),
                    "elements": inst.get("elements", []),
                })

    return samples


def _get_merger(model):
    """
    Match original baseline_comparison.py merger lookup.
    """
    for fn in [
        lambda m: m.model.model.visual.merger,
        lambda m: m.base_model.model.model.visual.merger,
        lambda m: m.model.visual.merger,
    ]:
        try:
            v = fn(model)
            if v is not None:
                return v
        except (AttributeError, TypeError):
            pass

    if hasattr(model, "_get_visual_merger"):
        return model._get_visual_merger()

    return None


@torch.inference_mode()
def predict(model, processor, image, question, elements, device, max_new_tokens, original_size=None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    # ========================================================
    # Original-paper GT-elements -> FFEM region injection
    # ========================================================
    merger = _get_merger(model)

    if merger is None:
        raise RuntimeError("Cannot locate visual merger for GT-region injection")

    if elements:
        merger.norm_elements = generate_region_labels_normalized(
            elements,
            original_image_size=(original_size if original_size is not None
                                 else (image.width, image.height)),
        )
        merger.region_labels = None
    else:
        merger.norm_elements = None
        merger.region_labels = None

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    # Prevent GT region leaking into the next QA
    merger.norm_elements = None
    merger.region_labels = None

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_len:]

    pred = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return pred.strip()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda:0")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_test_samples(args.test_dir)

    print("=" * 80)
    print("ThangkaVLM unified test evaluation")
    print("=" * 80)
    print("Test QA count:", len(samples))

    if args.limit > 0:
        samples = samples[:args.limit]
        print("Smoke-test limit:", len(samples))

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    print("Loading base model...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        args.adapter_path,
        is_trainable=False,
    )

    print("Loading FFEM weights...")
    ffem_state = torch.load(args.ffem_path, map_location="cpu")

    model_state = model.state_dict()

    loaded = []
    missing = []

    for k, v in ffem_state.items():
        if k in model_state:
            model_state[k].copy_(
                v.to(
                    device=model_state[k].device,
                    dtype=model_state[k].dtype,
                )
            )
            loaded.append(k)
        else:
            missing.append(k)

    print(f"FFEM tensors loaded: {len(loaded)}/{len(ffem_state)}")

    if missing:
        print("Missing FFEM keys:")
        for k in missing:
            print("  ", k)

    if len(loaded) != len(ffem_state):
        raise RuntimeError(
            f"FFEM loading incomplete: {len(loaded)}/{len(ffem_state)}"
        )

    # Ensure FFEM is enabled at runtime.
    try:
        model.base_model.model.set_ffem_enabled(True)
    except Exception:
        try:
            model.base_model.model.ffem.ffem_enabled = True
        except Exception as e:
            raise RuntimeError(f"无法启用FFEM: {e}")

    model.eval()

    rows = []
    correct = 0

    for i, sample in enumerate(samples, 1):
        image = Image.open(sample["image_path"]).convert("RGB")

        original_size = image.size   # (W,H) resize 之前
        image = resize_for_eval(image, args.image_max_side)

        pred = predict(
            model,
            processor,
            image,
            sample["question"],
            sample["elements"],
            device,
            args.max_new_tokens,
        )

        ngt = normalize_answer(sample["ground_truth"])
        npred = normalize_answer(pred)
        is_correct = int(is_correct_answer(pred, sample["ground_truth"]))

        correct += is_correct

        row = {
            **sample,
            "prediction": pred,
            "normalized_gt": ngt,
            "normalized_prediction": npred,
            "correct": is_correct,
            "hallucination": "",
            "hallucination_type": "",
            "reviewer_note": "",
        }

        rows.append(row)

        print(
            f"[{i:04d}/{len(samples):04d}] "
            f"type={sample['question_type']} | "
            f"GT={sample['ground_truth']} | "
            f"PRED={pred} | "
            f"{'✓' if is_correct else '✗'}"
        )

    jsonl_path = out_dir / "predictions.jsonl"
    csv_path = out_dir / "predictions.csv"
    review_path = out_dir / "review_hallucination.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = list(rows[0].keys())

    for path in [csv_path, review_path]:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    acc = correct / len(rows)

    type_stats = {}

    for row in rows:
        t = row["question_type"]

        if t not in type_stats:
            type_stats[t] = {"n": 0, "correct": 0}

        type_stats[t]["n"] += 1
        type_stats[t]["correct"] += row["correct"]

    for t, x in type_stats.items():
        x["accuracy"] = x["correct"] / x["n"]

    metrics = {
        "n": len(rows),
        "correct": correct,
        "normalized_exact_accuracy": acc,
        "hallucination_rate": None,
        "question_type_metrics": type_stats,
        "evaluation_protocol": {
            "generation": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "image_max_side": args.image_max_side,
            "normalization": "NFKC + lowercase + punctuation/space removal + yes/no canonicalization + target containment with negation guard",
            "hallucination": "pending independent review; incorrect != hallucination",
        },
    }

    (out_dir / "auto_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("N        =", len(rows))
    print("Correct  =", correct)
    print(f"Accuracy = {acc:.4%}")
    print("Outputs  =", out_dir)


if __name__ == "__main__":
    main()
