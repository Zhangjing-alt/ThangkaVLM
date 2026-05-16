# ffem.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FFEM(nn.Module):
    """
    Frequency-aware Feature Enhancement Module (FFEM)
    最终修复版：
    1. 适配 Qwen3-VL 视觉特征维度（[B, 392, D]），不再强制对齐文本序列长度
    2. 修复 region_labels 维度匹配问题，支持动态视觉特征长度
    3. 优化设备/ dtype 对齐逻辑，兼容 bf16/fp16 训练
    4. 修复 SRE 阶段 topk 计算逻辑，适配任意视觉特征长度
    """
    def __init__(
        self,
        embed_dim: int,
        num_regions: int = 16,
        topk_ratio: float = 0.25,
        ff_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_regions = num_regions
        self.topk_ratio = topk_ratio
        self.ff_hidden_dim = ff_hidden_dim if ff_hidden_dim is not None else embed_dim

        # ========== Stage 1: Multi-Scale Frequency Representation (MSFR) ==========
        # 适配 1D 视觉特征序列（Conv1d 更适合）
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.conv3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim)
        self.msfr_fusion = nn.Linear(embed_dim * 3, embed_dim)

        # ========== Stage 2: Semantic Region Extraction (SRE) ==========
        self.region_proj = nn.Linear(embed_dim, num_regions)
        self.sre_norm = nn.LayerNorm(embed_dim)

        # ========== Stage 3: Semantic-Spatial Attention Mixing (SSAM) ==========
        self.spatial_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.semantic_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.ssam_fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.ssam_norm = nn.LayerNorm(embed_dim)

        self.semantic_k_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.semantic_v_proj = nn.Linear(embed_dim * 2, embed_dim)

        # ========== FFN & Output ==========
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_hidden_dim),
            nn.GELU(),
            nn.Linear(self.ff_hidden_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

        # 记录上一次使用的设备，避免重复迁移日志刷屏
        self._last_device = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _ensure_on_device(self, device: torch.device):
        """
        将 FFEM 所有参数/缓冲区动态迁移到目标设备。
        以输入 tensor 的设备为基准，而不是以参数设备为基准，
        从而彻底解决 accelerate 多卡环境下设备漂移问题。
        只在设备真正发生变化时才执行迁移，避免每次 forward 都有开销。
        """
        if self._last_device == device:
            return
        current_param_device = self.conv1.weight.device
        if current_param_device != device:
            print(f"    [FFEM Device] Moving FFEM params from {current_param_device} to {device}")
            # to() 对整个 Module 生效，包含所有子模块的参数和 buffer
            # 注意：不改变 dtype，只迁移设备
            super().to(device)
        self._last_device = device

    def forward(
        self,
        visual_tokens: torch.Tensor,
        region_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        forward: 接受 3D 张量输入 (B, N, D)，N 为视觉特征长度（如 392）
        核心修复：
        1. 动态适配视觉特征长度 N，不再强制要求 1024
        2. region_labels 自动截断/填充到当前视觉特征长度
        3. 优化 topk 计算逻辑，避免极端值
        """
        if visual_tokens.dim() != 3:
            raise ValueError(
                f"FFEM expects 3D input (B, N, D), got shape={tuple(visual_tokens.shape)}."
            )

        # ========== 设备对齐：以输入设备为基准，迁移 FFEM 参数 ==========
        input_device = visual_tokens.device
        self._ensure_on_device(input_device)

        # region_labels 对齐到输入设备 + 动态适配视觉特征长度
        B, N, D = visual_tokens.shape
        if region_labels is not None:
            # 对齐设备
            if region_labels.device != input_device:
                region_labels = region_labels.to(input_device)
            
            # 核心修复1：动态适配视觉特征长度 N
            if region_labels.shape[1] != N:
                # 截断过长的 region_labels
                if region_labels.shape[1] > N:
                    region_labels = region_labels[:, :N]
                # 填充过短的 region_labels（补0）
                else:
                    pad = torch.zeros(
                        (B, N - region_labels.shape[1]), 
                        device=input_device, 
                        dtype=region_labels.dtype
                    )
                    region_labels = torch.cat([region_labels, pad], dim=1)
            
            # 调试信息：打印 region_labels 状态
            non_zero_count = region_labels.nonzero().size(0)
            print(f"    [FFEM Debug] region_labels shape: {region_labels.shape}, "
                  f"non-zero count: {non_zero_count}")
            
            # 修复 region_labels 全0问题：自动标记中心区域
            if non_zero_count == 0:
                print("    [FFEM Warning] region_labels is all zero, auto-mark center region")
                center_start = max(0, N // 2 - 10)
                center_end = min(N, N // 2 + 10)
                region_labels[:, center_start:center_end] = 1
        else:
            print("    [FFEM Debug] region_labels is None")

        # ========== 基础配置 ==========
        residual = visual_tokens  # (B, N, D)
        dtype = visual_tokens.dtype

        # ========== Stage 1: MSFR ==========
        tokens_1d = visual_tokens.permute(0, 2, 1)  # (B, D, N)

        feat1 = F.gelu(self.conv1(tokens_1d))  # (B, D, N)
        feat2 = F.gelu(self.conv2(tokens_1d))  # (B, D, N)
        feat3 = F.gelu(self.conv3(tokens_1d))  # (B, D, N)

        msfr_concat = torch.cat([feat1, feat2, feat3], dim=1)  # (B, 3D, N)
        msfr_concat = msfr_concat.permute(0, 2, 1)             # (B, N, 3D)
        msfr_out = self.msfr_fusion(msfr_concat)               # (B, N, D)
        msfr_out = msfr_out + residual                         # 残差 (B, N, D)
        msfr_out = msfr_out.to(dtype)

        # ========== Stage 2: SRE ==========
        region_scores = self.region_proj(msfr_out)             # (B, N, num_regions)
        region_probs = F.softmax(region_scores, dim=-1)        # (B, N, num_regions)

        # 核心修复2：优化 topk 计算，适配任意 N
        topk = max(min(int(N * self.topk_ratio), N), 8)  # 至少取8个，避免极端小值
        topk = min(topk, 64)  # 最多取64个，避免内存溢出

        topk_probs, topk_indices = torch.topk(region_probs, k=topk, dim=1)  # (B, topk, num_regions)

        batch_indices = torch.arange(B, device=input_device)[:, None, None]  # (B, 1, 1)
        topk_tokens = msfr_out[batch_indices, topk_indices, :]          # (B, topk, D)

        sre_out = self.sre_norm(topk_tokens)  # (B, topk, D)
        sre_out = sre_out.to(dtype)

        # 还原到原长度
        sre_full = msfr_out.clone()
        # 处理 batch_indices 维度匹配问题
        if topk_indices.shape[1] > 0:
            sre_full[batch_indices.expand(-1, topk, -1), topk_indices, :] = sre_out
        sre_out = sre_full + msfr_out  # 残差 (B, N, D)
        sre_out = sre_out.to(dtype)

        # ========== Stage 3: SSAM ==========
        spatial_attn_out, _ = self.spatial_attn(sre_out, sre_out, sre_out)  # (B, N, D)
        spatial_attn_out = spatial_attn_out.to(dtype)

        if region_labels is not None:
            if region_labels.dim() == 2:
                # 修复 dtype 问题：one_hot 转输入 dtype
                region_labels_onehot = F.one_hot(
                    region_labels.long(), num_classes=self.num_regions
                ).to(dtype)  # (B, N, num_regions)
            elif region_labels.dim() == 3:
                region_labels_onehot = region_labels.to(dtype)
            else:
                raise ValueError(
                    f"region_labels must be 2D (B, N) or 3D (B, N, num_regions), "
                    f"got shape={tuple(region_labels.shape)}"
                )

            # 修复分母为0问题
            region_sum = region_labels_onehot.sum(1, keepdim=True).transpose(1, 2) + 1e-6
            region_centers = torch.bmm(region_labels_onehot.transpose(1, 2), sre_out)  # (B, num_regions, D)
            region_centers = region_centers / region_sum

            region_feat = torch.bmm(region_labels_onehot, region_centers)  # (B, N, D)

            semantic_kv_concat = torch.cat([sre_out, region_feat], dim=-1)  # (B, N, 2D)
            semantic_k = self.semantic_k_proj(semantic_kv_concat).to(dtype)
            semantic_v = self.semantic_v_proj(semantic_kv_concat).to(dtype)

            semantic_attn_out, _ = self.semantic_attn(sre_out, semantic_k, semantic_v)
        else:
            semantic_attn_out, _ = self.semantic_attn(sre_out, sre_out, sre_out)

        semantic_attn_out = semantic_attn_out.to(dtype)

        ssam_concat = torch.cat([spatial_attn_out, semantic_attn_out], dim=-1)  # (B, N, 2D)
        ssam_fused = self.ssam_fusion(ssam_concat)                               # (B, N, D)
        ssam_out = self.ssam_norm(ssam_fused + sre_out).to(dtype)               # (B, N, D)

        # ========== FFN & Final Output ==========
        ffn_out = self.ffn(ssam_out)                                             # (B, N, D)
        enhanced_tokens = self.output_norm(ffn_out + ssam_out).to(dtype)        # (B, N, D)

        # 调试信息
        with torch.no_grad():
            diff = (enhanced_tokens - visual_tokens).abs().mean().item()
            print(f"    [FFEM Debug] input mean: {visual_tokens.mean().item():.4f}, "
                  f"output mean: {enhanced_tokens.mean().item():.4f}, "
                  f"mean abs diff: {diff:.6f}")

        # 最终校验：输出维度必须和输入一致
        if enhanced_tokens.shape != (B, N, D):
            raise ValueError(
                f"FFEM output shape mismatch: expected {(B, N, D)}, "
                f"got {tuple(enhanced_tokens.shape)}"
            )

        return enhanced_tokens


# 测试代码（适配 Qwen3-VL 真实维度）
if __name__ == "__main__":
    embed_dim = 4096  # Qwen3-VL 视觉特征维度
    num_regions = 16
    model = FFEM(embed_dim=embed_dim, num_regions=num_regions)

    # 测试 Qwen3-VL 真实维度 (B=2, N=392, D=4096)
    model = model.to(torch.bfloat16)
    visual_tokens = torch.randn(2, 392, 4096, dtype=torch.bfloat16).cuda()
    # 模拟 region_labels 长度不匹配（1024→392）
    region_labels = torch.randint(0, num_regions, (2, 1024)).cuda()

    output = model(visual_tokens, region_labels)
    print(f"Input  shape: {visual_tokens.shape}, dtype: {visual_tokens.dtype}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    assert output.shape == (2, 392, 4096), "❌ 维度适配失败！"
    assert output.dtype == torch.bfloat16, "❌ dtype 不匹配！"
    print("✅ FFEM Qwen3-VL 维度适配测试通过！")

    # 测试 region_labels 全0场景
    region_labels_zero = torch.zeros((2, 392), dtype=torch.long).cuda()
    output = model(visual_tokens, region_labels_zero)
    print(f"\nTest region_labels all zero:")
    print(f"Output shape: {output.shape}")
    print("✅ FFEM region_labels 全0自动修复测试通过！")