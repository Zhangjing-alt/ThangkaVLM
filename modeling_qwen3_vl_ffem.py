# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


# ============================================================
# 工具函数：生成归一化 region 标注
# ============================================================

def generate_region_labels_normalized(
    elements: List[Dict],
    original_image_size: Optional[tuple] = None,
) -> Optional[List[Dict]]:
    if not elements:
        return None

    norm_elements = []
    for region_id, element in enumerate(elements, start=1):
        if "bbox" not in element:
            continue

        bbox = element["bbox"]
        is_pixel = any(v > 1 for v in bbox)

        if is_pixel:
            if original_image_size is None:
                raise RuntimeError(
                    "[region_labels] bbox 为像素坐标但未提供 original_image_size。"
                    "必须传入 resize 之前的原图 (W,H)，否则区域标签会错位。")
            else:
                w, h = original_image_size
            x1, y1, x2, y2 = (
                bbox[0] / w,
                bbox[1] / h,
                bbox[2] / w,
                bbox[3] / h,
            )
        else:
            x1, y1, x2, y2 = bbox

        x1 = float(max(0.0, min(1.0, x1)))
        y1 = float(max(0.0, min(1.0, y1)))
        x2 = float(max(0.0, min(1.0, x2)))
        y2 = float(max(0.0, min(1.0, y2)))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        norm_elements.append(
            {
                "region_id": region_id,
                "bbox_norm": (x1, y1, x2, y2),
                "name": element.get("name_zh", element.get("label", f"Region{region_id}")),
            }
        )

    return norm_elements if norm_elements else None


def build_region_labels_for_tokens(
    norm_elements: List[Dict],
    num_tokens: int,
    device,
) -> torch.Tensor:
    """
    Revision fix: 复用训练路径的实现（region_labels_utils），
    保证训练与推理的 region 网格映射、重叠框处理顺序完全一致。
    重叠处理：按面积从大到小绘制，小区域覆盖大区域。
    """
    from region_labels_utils import build_region_labels_on_merged_grid

    if not norm_elements:
        return torch.zeros(num_tokens, dtype=torch.long, device=device)

    boxes, ids = [], []
    for elem in norm_elements:
        boxes.append(list(elem["bbox_norm"]))
        ids.append(int(elem["region_id"]))

    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise RuntimeError(
            f"[region_labels] num_tokens={num_tokens} 非完全平方数，"
            f"无法推断方形 token 网格；请改为显式传入 image_grid_thw。")

    grid_thw = torch.tensor([1, side * 2, side * 2])
    lab = build_region_labels_on_merged_grid(boxes, grid_thw, merge_size=2)

    # 把 1..K 的连续编号映射回原始 region_id
    out = torch.zeros_like(lab)
    for pos, rid in enumerate(ids, start=1):
        out[lab == pos] = rid
    return out.to(device)


def _legacy_build_region_labels_for_tokens(
    norm_elements: List[Dict],
    num_tokens: int,
    device,
) -> torch.Tensor:
    grid_size = int(math.isqrt(num_tokens))
    if grid_size * grid_size < num_tokens:
        grid_size += 1

    labels = torch.zeros(num_tokens, dtype=torch.long, device=device)

    for elem in norm_elements:
        x1_n, y1_n, x2_n, y2_n = elem["bbox_norm"]
        rid = int(elem["region_id"])

        # Match corrected training-time bbox -> post-merger grid mapping exactly
        x1 = max(
            0,
            min(
                grid_size - 1,
                int(math.floor(x1_n * grid_size)),
            ),
        )
        y1 = max(
            0,
            min(
                grid_size - 1,
                int(math.floor(y1_n * grid_size)),
            ),
        )

        x2 = max(
            x1 + 1,
            min(
                grid_size,
                int(math.ceil(x2_n * grid_size)),
            ),
        )
        y2 = max(
            y1 + 1,
            min(
                grid_size,
                int(math.ceil(y2_n * grid_size)),
            ),
        )

        for y in range(y1, y2):
            for x in range(x1, x2):
                idx = y * grid_size + x
                if idx < num_tokens:
                    labels[idx] = rid

    return labels


# ============================================================
# robust_get_elements：兼容多种字段名的 elements 提取
# ============================================================

ELEMENT_FIELD_CANDIDATES = [
    "elements", "normalized_elements", "norm_elements",
    "regions", "bboxes", "components", "objects",
    "annotations", "parts", "labels",
]

def robust_get_elements(inst: dict) -> list:
    """
    从 JSON 实例中提取 elements，兼容多种字段名。
    在 load_test_samples 中替换原有的 inst.get("elements") or ... 逻辑。
    """
    for field in ELEMENT_FIELD_CANDIDATES:
        val = inst.get(field)
        if val and isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], dict) and (
                "bbox"    in val[0] or
                "label"   in val[0] or
                "name"    in val[0] or
                "name_zh" in val[0]
            ):
                return val
    return []


# ============================================================
# FFEM 模块
# ============================================================

class FFEM(nn.Module):
    def __init__(
        self,
        embed_dim: int = 4096,
        num_regions: int = 16,
        topk_ratio: float = 0.25,
        debug: bool = False,
        disable_rce: bool = False,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_regions = num_regions
        self.topk_ratio  = topk_ratio
        self.debug       = debug
        self.ffem_enabled = True
        self.disable_rce = disable_rce

        if self.debug:
            print(f"[FFEM Init] disable_rce={self.disable_rce}")

        # ===== MSFR =====
        self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, embed_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(1, embed_dim, kernel_size=7, padding=3)
        self.msfr_fusion = nn.Linear(embed_dim * 3, embed_dim)

        # ===== RCE: Region Context Enhancement =====
        self.region_proj = nn.Linear(embed_dim, num_regions)

        # ===== SSAM =====
        self.semantic_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=max(1, embed_dim // 128),
            batch_first=True,
        )
        self.semantic_k_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.semantic_v_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=max(1, embed_dim // 128),
            batch_first=True,
        )

        self.sre_norm    = nn.LayerNorm(embed_dim)
        self.ssam_norm   = nn.LayerNorm(embed_dim)
        self.ssam_fusion = nn.Linear(embed_dim * 2, embed_dim)

        # ===== FFN & 输出 =====
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

        # ===== 残差缩放 =====
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        # RCE 运行时开关（返修新增）
        # ===== 路线2 诊断开关 =====
        self.residual_max     = 0.3    # D2: 残差上限
        self.region_to_query  = False  # D4: region 直接注入 query
        self.region_query_w   = 0.5    # D4 权重
        self.export_region_ctx = False # D3: 导出中间量供 alignment loss
        self._last_region_ctx = None
        self._last_region_labels = None
        self._last_enhanced = None
        self.rce_strict = True          # 长度不匹配 -> 直接报错
        self.rce_allow_none = True      # 允许显式 no-label 消融
        self._rce_fallback_count = 0

        self._weight_logged = False

    # ----------------------------------------------------------
    # MSFR
    # ----------------------------------------------------------
    def _msfr(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        MAX_TOKENS = 1024

        if N > MAX_TOKENS:
            idx = torch.linspace(0, N - 1, MAX_TOKENS,
                                 dtype=torch.long, device=x.device)
            x_sample = x[:, idx, :]
            sample_len = MAX_TOKENS
        else:
            x_sample = x
            sample_len = N

        x_1d = x_sample.reshape(B * sample_len, 1, D)

        c1 = self.conv1(x_1d).mean(dim=-1)
        c2 = self.conv2(x_1d).mean(dim=-1)
        c3 = self.conv3(x_1d).mean(dim=-1)

        msfr = torch.cat([c1, c2, c3], dim=-1)
        msfr = self.msfr_fusion(msfr)
        msfr = msfr.view(B, sample_len, D)

        if sample_len < N:
            msfr = msfr.permute(0, 2, 1)
            msfr = F.interpolate(msfr, size=N, mode='linear', align_corners=False)
            msfr = msfr.permute(0, 2, 1)

        return msfr

    # ----------------------------------------------------------
    # RCE：区域上下文编码
    # ----------------------------------------------------------
    def _build_region_context(
        self,
        x: torch.Tensor,
        region_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Batch-aware RCE.

        x:
            (B, N, D)

        region_labels:
            None
            or (N,) when B == 1
            or (B, N)

        Each sample is processed independently.
        No region pooling or gating crosses batch boundaries.
        """
        B, N, D = x.shape

        # ------------------------------------------------------
        # No-label mode
        # ------------------------------------------------------
        if region_labels is None:
            if (
                getattr(self, "rce_strict", False)
                and not getattr(self, "rce_allow_none", True)
            ):
                raise RuntimeError(
                    "[RCE] strict 模式下 region_labels 不可为 None"
                )

            self._rce_fallback_count = (
                getattr(self, "_rce_fallback_count", 0) + 1
            )

            global_ctx = x.mean(dim=1, keepdim=True)
            return global_ctx.expand(B, N, D)

        # ------------------------------------------------------
        # Normalize labels to (B, N)
        # ------------------------------------------------------
        labels = region_labels.to(x.device)

        if labels.ndim == 1:
            if B != 1:
                raise RuntimeError(
                    "[RCE] batch_size > 1 时 region_labels "
                    "必须保留为 (B, N)，禁止展平。"
                )

            if labels.numel() != N:
                raise RuntimeError(
                    f"[RCE] region_labels 长度 "
                    f"{labels.numel()} != token 数 {N}"
                )

            labels = labels.view(1, N)

        elif labels.ndim == 2:
            if tuple(labels.shape) != (B, N):
                raise RuntimeError(
                    f"[RCE] region_labels shape "
                    f"{tuple(labels.shape)} != expected {(B, N)}"
                )

        else:
            raise RuntimeError(
                f"[RCE] unsupported region_labels ndim="
                f"{labels.ndim}"
            )

        # ------------------------------------------------------
        # Batch-independent region context
        # ------------------------------------------------------
        region_ctx = torch.zeros_like(x)
        region_logits = self.region_proj(x)

        for b in range(B):
            xb = x[b:b + 1]
            lb = labels[b]

            global_ctx_b = xb.mean(
                dim=1,
                keepdim=True,
            )

            max_region_id = (
                int(lb.max().item())
                if lb.numel() > 0
                else 0
            )

            # No foreground region in this sample
            if max_region_id <= 0:
                region_ctx[b:b + 1] = global_ctx_b.expand(
                    1, N, D
                )
                continue

            for rid in range(
                1,
                max_region_id + 1,
            ):
                mask = lb == rid

                if not torch.any(mask):
                    continue

                tokens_r = xb[:, mask, :]

                pooled = tokens_r.mean(
                    dim=1,
                    keepdim=True,
                )

                region_idx = min(
                    rid - 1,
                    self.num_regions - 1,
                )

                gate = torch.sigmoid(
                    region_logits[
                        b:b + 1,
                        mask,
                        region_idx,
                    ].mean()
                ).view(1, 1, 1)

                pooled = pooled * gate

                region_ctx[
                    b:b + 1,
                    mask,
                    :
                ] = pooled.expand(
                    1,
                    int(mask.sum().item()),
                    D,
                )

            # Background uses this sample's own global context
            bg_mask = lb == 0

            if torch.any(bg_mask):
                region_ctx[
                    b:b + 1,
                    bg_mask,
                    :
                ] = global_ctx_b.expand(
                    1,
                    int(bg_mask.sum().item()),
                    D,
                )

        return region_ctx


    # ----------------------------------------------------------
    # forward
    # ----------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        region_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.ffem_enabled:
            return hidden_states

        orig_ndim = hidden_states.ndim
        if orig_ndim == 2:
            x = hidden_states.unsqueeze(0)
        elif orig_ndim == 3:
            x = hidden_states
        else:
            raise ValueError(f"Unsupported hidden_states ndim={hidden_states.ndim}")

        if self.debug and not self._weight_logged:
            w = self.conv1.weight
            print(f"[FFEM权重诊断] conv1 mean={w.mean():.6f}, std={w.std():.6f}")
            print(f"[FFEM权重诊断] residual_scale(sigmoid)="
                  f"{torch.sigmoid(self.residual_scale).item():.4f}")
            if w.std() < 0.005 or w.std() > 0.5:
                print("[FFEM权重诊断] ⚠️ std异常，权重可能未经充分训练！")
            self._weight_logged = True

        msfr = self._msfr(x)

        if self.disable_rce:
            region_ctx = torch.zeros_like(x)
        else:
            # Preserve batch boundaries.
            # (B, N) labels must never be flattened across samples.
            rl_for_rce = region_labels
            region_ctx = self._build_region_context(
                x,
                rl_for_rce,
            )

        kv_input = torch.cat([x, region_ctx], dim=-1)
        k = self.semantic_k_proj(kv_input)
        v = self.semantic_v_proj(kv_input)

        # D4: region context 直接参与 query
        if getattr(self, "region_to_query", False):
            q_in = x + self.region_query_w * region_ctx
        else:
            q_in = x

        sem_out, _ = self.semantic_attn(query=q_in, key=k, value=v, need_weights=False)
        sem_out = self.sre_norm(x + sem_out)

        spa_out, _ = self.spatial_attn(
            query=sem_out, key=sem_out, value=sem_out, need_weights=False)
        spa_out = self.ssam_norm(sem_out + spa_out)

        fused    = self.ssam_fusion(torch.cat([msfr, spa_out], dim=-1))
        ffn_out  = self.ffn(fused)
        enhanced = self.output_norm(fused + ffn_out)

        scale = torch.sigmoid(self.residual_scale) * getattr(self, "residual_max", 0.3)
        out   = x + scale * (enhanced - x)

        if getattr(self, "export_region_ctx", False):
            self._last_region_ctx    = region_ctx
            self._last_region_labels = rl_for_rce
            self._last_enhanced      = enhanced

        if self.debug:
            with torch.no_grad():
                diff = (out - x).abs().mean().item()
                print(
                    f"    [FFEM Debug] region_labels="
                    f"{'None' if region_labels is None else tuple(region_labels.shape)}"
                )
                print(
                    f"    [FFEM Debug] input mean: {x.mean().item():.4f}, "
                    f"output mean: {out.mean().item():.4f}, "
                    f"mean abs diff: {diff:.6f}, "
                    f"residual_scale(sigmoid): {torch.sigmoid(self.residual_scale).item():.4f}"
                )

        if orig_ndim == 2:
            out = out.squeeze(0)

        return out


# ============================================================
# RobustRCE：修复版区域上下文编码
# ============================================================

class RobustRCE:
    """
    替换 FFEM._build_region_context 的修复版本。
    修复：
      1) token 数不匹配时用最近邻插值对齐，而非静默回退 global context
      2) 引入 contrast_weight 使区域特征与全局特征形成对比，
         避免 Full FFEM 与 w/o RCE 差异过小
    用法（在消融评估时由 patch_ffem_for_ablation 自动注入）：
        model.base_model.model.ffem._build_region_context = RobustRCE(ffem)
    """

    def __init__(self, ffem_module: FFEM, contrast_weight: float = 0.5):
        self.ffem            = ffem_module
        self.contrast_weight = contrast_weight

    def __call__(
        self,
        x: torch.Tensor,
        region_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N, D = x.shape
        assert B == 1

        if region_labels is not None:
            region_labels = region_labels.reshape(-1).to(x.device)

        # 修复1：token 数不匹配时插值对齐
        if region_labels is not None and region_labels.numel() != N:
            old_len = region_labels.numel()
            rl_f = region_labels.float().unsqueeze(0).unsqueeze(0)
            rl_f = F.interpolate(rl_f, size=N, mode='nearest')
            region_labels = rl_f.squeeze().long()
            warnings.warn(
                f"[RobustRCE] token数不匹配，已插值对齐: {old_len} → {N}",
                stacklevel=2,
            )

        if region_labels is None or region_labels.numel() != N:
            global_ctx = x.mean(dim=1, keepdim=True)
            return global_ctx.expand(B, N, D)

        labels        = region_labels
        max_region_id = int(labels.max().item()) if labels.numel() > 0 else 0

        if max_region_id <= 0:
            global_ctx = x.mean(dim=1, keepdim=True)
            return global_ctx.expand(B, N, D)

        global_ctx    = x.mean(dim=1, keepdim=True)
        region_ctx    = torch.zeros_like(x)
        region_logits = self.ffem.region_proj(x)

        for rid in range(1, max_region_id + 1):
            mask = labels == rid
            if not torch.any(mask):
                continue
            tokens_r = x[:, mask, :]
            pooled   = tokens_r.mean(dim=1, keepdim=True)

            region_idx = min(rid - 1, self.ffem.num_regions - 1)
            gate = torch.sigmoid(
                region_logits[:, mask, region_idx].mean()
            ).view(1, 1, 1)

            pooled = pooled * gate

            # 修复2：对比全局特征，携带差异信息
            contrast_ctx = pooled - self.contrast_weight * global_ctx
            region_ctx[:, mask, :] = contrast_ctx.expand(
                1, mask.sum().item(), D)

        bg_mask = labels == 0
        if torch.any(bg_mask):
            region_ctx[:, bg_mask, :] = global_ctx.expand(
                1, bg_mask.sum().item(), D)

        return region_ctx


# ============================================================
# Qwen3VLWithFFEM
# ============================================================

class Qwen3VLWithFFEM(Qwen3VLForConditionalGeneration):
    config_class = Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)

        embed_dim = getattr(config, "ffem_embed_dim", None)
        if embed_dim is None:
            embed_dim = getattr(config, "hidden_size", 4096)

        num_regions  = getattr(config, "ffem_num_regions", 16)
        topk_ratio   = getattr(config, "ffem_topk_ratio", 0.25)
        enable_debug = getattr(config, "enable_ffem_debug", False)
        disable_rce  = getattr(config, "ffem_disable_rce", False)

        self.ffem = FFEM(
            embed_dim=embed_dim,
            num_regions=num_regions,
            topk_ratio=topk_ratio,
            debug=enable_debug,
            disable_rce=disable_rce,
        )

        self._ffem_hook_handle     = None
        self._ffem_pre_hook_handle = None
        self._register_ffem_hook()

        print(
            f"✅ FFEM模块已初始化：embed_dim={embed_dim}, "
            f"num_regions={num_regions}, topk_ratio={topk_ratio}"
        )

    # ----------------------------------------------------------
    # 工具
    # ----------------------------------------------------------
    def _get_visual_merger(self):
        if (hasattr(self, "model")
                and hasattr(self.model, "visual")
                and hasattr(self.model.visual, "merger")):
            return self.model.visual.merger
        if hasattr(self, "visual") and hasattr(self.visual, "merger"):
            return self.visual.merger
        return None

    def _get_current_ffem(self):
        """
        ✅ 核心修复：动态查找当前实际挂载的 ffem。
        PeftModel 包装后，self.ffem 经 __getattr__ 代理最终指向
        Qwen3VLWithFFEM.ffem（原始 FFEM），无法感知消融时的替换。
        改为优先从 base_model.model 路径直接取，保证拿到 AblatedFFEM。
        """
        # PeftModel 包装后的实际 core 路径
        try:
            return self.base_model.model.ffem
        except AttributeError:
            pass
        # 未包装时直接返回 self.ffem
        return self.ffem

    # ----------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------
    def _ffem_pre_hook(self, module, inputs):
        if not hasattr(module, "region_labels"):
            module.region_labels = None
        if not hasattr(module, "norm_elements"):
            module.norm_elements = None
        return None

    def _ffem_hook(self, module, args, result):
        if result is None:
            return result
        if not torch.is_tensor(result):
            return result

        # ✅ 每次动态取当前生效的 ffem（可能是 AblatedFFEM）
        current_ffem = self._get_current_ffem()

        if not getattr(current_ffem, "ffem_enabled", True):
            return result

        out        = result
        out_device = out.device
        out_dtype  = out.dtype
        num_tokens = out.shape[0] if out.ndim == 2 else out.shape[1]

        # ===== 一次性权重诊断 =====
        if not getattr(self, '_ffem_weight_logged', False):
            try:
                w  = current_ffem.conv1.weight
                rs = current_ffem.residual_scale
            except AttributeError:
                # AblatedFFEM 包了一层
                w  = current_ffem.ffem.conv1.weight
                rs = current_ffem.ffem.residual_scale
            print(f"[FFEM权重诊断] conv1 mean={w.mean():.6f}, std={w.std():.6f}")
            print(f"[FFEM权重诊断] residual_scale(sigmoid)="
                  f"{torch.sigmoid(rs).item():.4f}")
            self._ffem_weight_logged = True

        # ===== 构建 region_labels =====
        region_labels = None
        norm_elements = getattr(module, "norm_elements", None)

        if norm_elements is not None:
            try:
                region_labels = build_region_labels_for_tokens(
                    norm_elements, num_tokens, out_device)
            except Exception as e:
                warnings.warn(f"FFEM动态构建region_labels失败，回退无监督分支: {e}")
                region_labels = None
        else:
            rl = getattr(module, "region_labels", None)
            if rl is not None and torch.is_tensor(rl):
                rl_flat = rl.reshape(-1).to(out_device)
                if rl_flat.numel() == num_tokens:
                    region_labels = rl_flat
                else:
                    warnings.warn(
                        f"region_labels token数({rl_flat.numel()})与"
                        f" merger输出token数({num_tokens})不匹配，回退无监督分支。"
                    )

        # ===== 设备 / dtype 对齐 =====
        try:
            ffem_device = next(current_ffem.parameters()).device
        except StopIteration:
            ffem_device = out_device
        if ffem_device != out_device:
            current_ffem = current_ffem.to(device=out_device, dtype=out_dtype)
        else:
            current_ffem = current_ffem.to(dtype=out_dtype)

        # ===== 调用当前实际生效的 ffem =====
        try:
            out_ffem = current_ffem(out, region_labels=region_labels)
        except torch.cuda.OutOfMemoryError as e:
            warnings.warn(f"FFEM OOM（已跳过）: {e}")
            return result

        if out_ffem.device != out_device:
            out_ffem = out_ffem.to(out_device)
        if out_ffem.dtype != out_dtype:
            out_ffem = out_ffem.to(out_dtype)

        return out_ffem

    def _register_ffem_hook(self):
        merger = self._get_visual_merger()
        if merger is None:
            warnings.warn("未找到 model.visual.merger，FFEM hook 注册失败。")
            return

        for attr in ("_ffem_pre_hook_handle", "_ffem_hook_handle"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass
            setattr(self, attr, None)

        if not hasattr(merger, "region_labels"):
            merger.region_labels = None
        if not hasattr(merger, "norm_elements"):
            merger.norm_elements = None

        self._ffem_pre_hook_handle = merger.register_forward_pre_hook(self._ffem_pre_hook)
        self._ffem_hook_handle     = merger.register_forward_hook(self._ffem_hook)

        print("✅ FFEM hook 已注册到 model.visual.merger")

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------
    def enable_ffem_debug(self, enabled: bool = True):
        self.ffem.debug = enabled
        self.ffem._weight_logged = False

    def set_ffem_enabled(self, enabled: bool = True):
        self.ffem.ffem_enabled = enabled
        # 同步到 base_model.model.ffem（PeftModel 包装后的实际路径）
        try:
            self.base_model.model.ffem.ffem_enabled = enabled
        except AttributeError:
            pass
        print(f"    [FFEM Switch] ffem_enabled = {enabled}")

    def set_ffem_region_labels(self, region_labels: Optional[torch.Tensor]):
        merger = self._get_visual_merger()
        if merger is not None:
            merger.region_labels = region_labels
            merger.norm_elements = None

    def set_ffem_norm_elements(self, norm_elements: Optional[List[Dict]]):
        merger = self._get_visual_merger()
        if merger is not None:
            merger.norm_elements = norm_elements
            merger.region_labels = None

    def clear_ffem_labels(self):
        merger = self._get_visual_merger()
        if merger is not None:
            merger.region_labels = None
            merger.norm_elements = None

    # ----------------------------------------------------------
    # forward
    # ----------------------------------------------------------
    def forward(
        self,
        *args,
        region_labels: Optional[torch.Tensor] = None,
        norm_elements: Optional[List[Dict]] = None,
        **kwargs,
    ):
        merger = self._get_visual_merger()
        if merger is not None:
            if norm_elements is not None:
                merger.norm_elements = norm_elements
                merger.region_labels = None
            elif region_labels is not None:
                merger.region_labels = region_labels
                merger.norm_elements = None

        return super().forward(*args, **kwargs)


# ============================================================
# 模型注册
# ============================================================

def register_qwen3_vl_ffem():
    try:
        AutoConfig.register("qwen3_vl", Qwen3VLConfig, exist_ok=True)
    except Exception:
        pass

    try:
        AutoModelForImageTextToText.register(
            Qwen3VLConfig, Qwen3VLWithFFEM, exist_ok=True)
        print("✅ 已注册 AutoModelForImageTextToText: Qwen3VLConfig -> Qwen3VLWithFFEM")
    except TypeError:
        try:
            AutoModelForImageTextToText.register(Qwen3VLConfig, Qwen3VLWithFFEM)
            print("✅ 已注册 AutoModelForImageTextToText: Qwen3VLConfig -> Qwen3VLWithFFEM")
        except Exception as e:
            print(f"⚠️ AutoModelForImageTextToText.register 失败: {e}")
    except Exception as e:
        print(f"⚠️ 模型注册失败: {e}")


register_qwen3_vl_ffem()