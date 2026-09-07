#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoProcessor,
    set_seed,
    get_scheduler,
    TrainerCallback,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from model_registration import Qwen3VLWithFFEM
import thangka_dataset as _td
from thangka_dataset import ThangkaDataset


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3VL-FFEM 唐卡模型微调训练脚本")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--task_mode", type=str, default="mixed")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--image_max_side", type=int, default=448)
    p.add_argument("--user_max_tokens", type=int, default=256)
    p.add_argument("--assistant_max_tokens", type=int, default=256)
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--tune", type=str, default="lora_ffem",
                   choices=["full", "ffem", "lora_ffem"])
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--use_bbox", type=bool, default=True)
    p.add_argument("--augment_qa", type=bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--early_stop_patience", type=int, default=2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    return p.parse_args()


# ======================== 参数识别工具 ========================
def _is_ffem_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    if name.startswith("ffem.") or ".ffem." in name:
        return True
    if "base_model.model.ffem" in name:
        return True
    ffem_keywords = [
        "ffem.conv1", "ffem.conv2", "ffem.conv3",
        "ffem.semantic_attn", "ffem.spatial_attn",
        "ffem.region_proj", "ffem.msfr_fusion",
        "ffem.sre_norm", "ffem.ssam_fusion",
        "ffem.output_norm", "ffem.ffn",
        "ffem.semantic_k_proj", "ffem.semantic_v_proj",
        "ffem.residual_scale",
    ]
    return any(k in name for k in ffem_keywords)


def _is_lora_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    return any(k in name for k in ["lora_A", "lora_B", "lora_embedding"])


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_main_process() -> bool:
    return _get_rank() == 0


# ======================== PEFT后重注册FFEM hook ========================
def _repatch_ffem_hook(peft_model):
    """
    PEFT包装后重新注册FFEM forward hook。

    核心修复（相比旧版）：
    - merger输出是 2D (N_total, D)，batch内所有图片tokens拼接在一起
    - region_labels是 (B, L)，需展平→截断/填充到 N_total 后传入FFEM
    - 旧版在3D分支处理时有索引维度错误，已全部重写
    """
    # -------- 查找 merger 模块 --------
    merger = None
    merger_paths = [
        "base_model.model.model.visual.merger",
        "base_model.model.visual.merger",
        "model.model.visual.merger",
        "model.visual.merger",
    ]
    for path in merger_paths:
        try:
            merger = eval(f"peft_model.{path}")
            if merger is not None:
                break
        except (AttributeError, NameError):
            continue
    if merger is None:
        for name, module in peft_model.named_modules():
            if name.endswith("visual.merger"):
                merger = module
                break
    if merger is None:
        if _is_main_process():
            print("⚠️ _repatch_ffem_hook: 未找到merger模块，跳过hook注册")
        return

    # -------- 查找 FFEM 模块 --------
    ffem = None
    ffem_paths = ["base_model.model.ffem", "model.ffem", "ffem"]
    for path in ffem_paths:
        try:
            ffem = eval(f"peft_model.{path}")
            if ffem is not None:
                break
        except (AttributeError, NameError):
            continue
    if ffem is None:
        for name, module in peft_model.named_modules():
            if name.endswith(".ffem") or name == "ffem":
                ffem = module
                break
    if ffem is None:
        if _is_main_process():
            print("⚠️ _repatch_ffem_hook: 未找到ffem模块，跳过hook注册")
        return

    # 清空旧 hook，防止重复注册
    merger._forward_hooks.clear()
    merger._forward_pre_hooks.clear()

    # -------- pre_hook：梯度修复 --------
    def _pre_hook(module, args):
        new_args = []
        for x in args:
            if isinstance(x, torch.Tensor) and not x.requires_grad:
                x = x.detach().requires_grad_(True)
            new_args.append(x)
        return tuple(new_args)

    # -------- forward hook：核心修复 --------
    def _ffem_hook(module, input, output):
        """
        修复要点：
        1. 训练时merger输出是 (N_total, D) 2D tensor
           N_total = batch_size × tokens_per_image（所有图片tokens拼接）
        2. region_labels是 (B, L)，展平后截断/填充到 N_total
        3. 推理时可能是 (B, N, D) 3D tensor，同样正确处理
        4. 对齐设备和dtype，避免设备漂移
        """
        region_labels = getattr(module, 'region_labels', None)

        try:
            if output.dim() == 2:
                # ===== 主路径（训练）：merger输出 (N_total, D) =====
                N_total, D = output.shape
                out_3d = output.unsqueeze(0)  # → (1, N_total, D)

                if region_labels is not None:
                    # (B, L) → 展平为 (B*L,) → 截断/填充到 N_total → (1, N_total)
                    rl_flat = region_labels.reshape(-1).to(output.device)
                    if rl_flat.shape[0] > N_total:
                        rl_flat = rl_flat[:N_total]
                    elif rl_flat.shape[0] < N_total:
                        pad = torch.zeros(
                            N_total - rl_flat.shape[0],
                            device=output.device,
                            dtype=rl_flat.dtype,
                        )
                        rl_flat = torch.cat([rl_flat, pad])
                    rl_for_ffem = rl_flat.unsqueeze(0)  # (1, N_total)
                else:
                    rl_for_ffem = None

                # dtype 对齐
                if next(ffem.parameters()).dtype != output.dtype:
                    ffem.to(dtype=output.dtype)

                enhanced = ffem(out_3d, region_labels=rl_for_ffem)  # (1, N_total, D)
                result = enhanced.squeeze(0)                          # (N_total, D)

                # 确保设备和dtype与原输出一致
                if result.device != output.device:
                    result = result.to(output.device)
                if result.dtype != output.dtype:
                    result = result.to(output.dtype)
                return result

            elif output.dim() == 3:
                # ===== 推理路径：merger输出 (B, N, D) =====
                B, N, D = output.shape

                if region_labels is not None:
                    rl = region_labels.to(output.device)
                    # 统一展平后截断/填充到 N，再reshape回 (1, N)
                    rl_flat = rl.reshape(-1)
                    if rl_flat.shape[0] > N:
                        rl_flat = rl_flat[:N]
                    elif rl_flat.shape[0] < N:
                        pad = torch.zeros(
                            N - rl_flat.shape[0],
                            device=output.device,
                            dtype=rl_flat.dtype,
                        )
                        rl_flat = torch.cat([rl_flat, pad])
                    rl_for_ffem = rl_flat.unsqueeze(0)  # (1, N)
                else:
                    rl_for_ffem = None

                # dtype 对齐
                if next(ffem.parameters()).dtype != output.dtype:
                    ffem.to(dtype=output.dtype)

                result = ffem(output, region_labels=rl_for_ffem)

                if result.device != output.device:
                    result = result.to(output.device)
                if result.dtype != output.dtype:
                    result = result.to(output.dtype)
                return result

            else:
                # 不支持的维度，透传
                if _is_main_process():
                    print(f"⚠️ FFEM hook: 不支持的输出维度 {output.dim()}D，透传原始输出")
                return output

        except Exception as e:
            if _is_main_process():
                out_shape = tuple(output.shape)
                rl_shape = tuple(region_labels.shape) if region_labels is not None else "None"
                print(f"⚠️ FFEM hook执行错误: {e}")
                print(f"    输出维度: {out_shape}, region_labels维度: {rl_shape}")
            return output  # fallback：透传原始输出

    merger.register_forward_pre_hook(_pre_hook)
    merger.register_forward_hook(_ffem_hook)

    if _is_main_process():
        print("✅ FFEM hook已重新注册（梯度修复 + region_labels展平对齐 + 2D/3D维度适配）")


# ======================== 微调模式配置 ========================
def apply_tuning_mode(args, model):
    if args.tune == "full":
        for p in model.parameters():
            p.requires_grad = True
        _print_trainable(model)
        return model

    elif args.tune == "ffem":
        for n, p in model.named_parameters():
            p.requires_grad = _is_ffem_name(n)
        _print_trainable(model)
        return model

    elif args.tune == "lora_ffem":
        from peft import LoraConfig, get_peft_model

        for p in model.parameters():
            p.requires_grad = False

        target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

        for n, p in model.named_parameters():
            if _is_ffem_name(n):
                p.requires_grad = True

        _repatch_ffem_hook(model)
        _print_trainable(model)
        return model

    else:
        raise ValueError(f"不支持的微调模式: {args.tune}")


def _print_trainable(model):
    if not _is_main_process():
        return
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n✅ 可训练参数统计:")
    print(f"   可训练: {trainable:,} | 总数: {total:,} | 占比: {100.*trainable/max(total,1):.4f}%")


def get_parameter_groups(args, model):
    lora_params, ffem_params, other_params = [], [], []

    if _is_main_process():
        print("\n" + "="*80)
        print("🔍 参数分组详细检查")
        print("="*80)

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_ffem_name(n):
            ffem_params.append(p)
            if _is_main_process():
                print(f"[FFEM] {n}")
        elif _is_lora_name(n):
            lora_params.append(p)
            if _is_main_process() and len(lora_params) <= 3:
                print(f"[LoRA] {n}")
        else:
            other_params.append(p)
            if _is_main_process():
                print(f"[其他] {n}")

    if _is_main_process():
        print(f"\n📊 参数统计:")
        print(f"  LoRA参数: {len(lora_params)} 个, 总量: {sum(p.numel() for p in lora_params):,}")
        print(f"  FFEM参数: {len(ffem_params)} 个, 总量: {sum(p.numel() for p in ffem_params):,}")
        print(f"  其他参数: {len(other_params)} 个, 总量: {sum(p.numel() for p in other_params):,}")
        # 更新预期数量为35（含新增的residual_scale）
        if len(ffem_params) not in (34, 35):
            print(f"\n⚠️ FFEM参数数量异常！预期34或35，实际{len(ffem_params)}（仅警告，不终止训练）")
        print("="*80 + "\n")

    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": args.learning_rate})
        if _is_main_process():
            print(f"✅ LoRA学习率: {args.learning_rate:.2e}")
    if ffem_params:
        groups.append({"params": ffem_params, "lr": args.learning_rate * 2})
        if _is_main_process():
            print(f"✅ FFEM学习率: {args.learning_rate * 2:.2e} (LoRA的2倍)")
    if other_params:
        groups.append({"params": other_params, "lr": args.learning_rate})
        if _is_main_process():
            print(f"✅ 其他参数学习率: {args.learning_rate:.2e}")
    if not groups:
        raise RuntimeError("❌ 未找到任何可训练参数！")
    return groups


# ======================== 早停回调 ========================
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 2):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.stopped_epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        loss = metrics.get("eval_loss")
        if loss is None:
            return
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if _is_main_process():
                print(f"\n⚠️ 早停计数器: {self.counter}/{self.patience} "
                      f"(当前loss={loss:.4f}, 最优={self.best_loss:.4f})")
            if self.counter >= self.patience:
                if _is_main_process():
                    print(f"\n🛑 早停触发：验证集loss连续{self.patience}轮未下降")
                self.stopped_epoch = state.epoch
                control.should_training_stop = True


# ======================== FFEM 参数变化验证回调 ========================
class FFEMGradientCheckCallback(TrainerCallback):
    def __init__(self, check_after_steps: int = 10):
        self.check_after_steps = check_after_steps
        self._initial_params = {}
        self._checked = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or not _is_main_process():
            return
        for name, param in model.named_parameters():
            if "ffem" in name and param.requires_grad:
                self._initial_params[name] = param.detach().cpu().clone()
        print(f"✅ [参数变化检查] 已记录 {len(self._initial_params)} 个FFEM参数初始值")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self._checked or state.global_step < self.check_after_steps or not _is_main_process():
            return
        self._checked = True

        print("\n" + "="*80)
        print(f"🔍 [参数变化检查] 第{state.global_step}步后FFEM参数变化情况")
        print("="*80)

        changed = unchanged = 0
        for name, param in model.named_parameters():
            if "ffem" not in name or name not in self._initial_params:
                continue
            initial = self._initial_params[name]
            current = param.detach().cpu()
            diff = (current - initial).abs().mean().item()
            rel = diff / (initial.abs().mean().item() + 1e-8)
            if diff > 1e-7:
                print(f"  ✅ {name}: 绝对变化={diff:.2e}, 相对变化={rel*100:.4f}%")
                changed += 1
            else:
                print(f"  ❌ {name}: 几乎未变化 (diff={diff:.2e})")
                unchanged += 1

        print(f"\n📊 统计: {changed} 个参数已更新, {unchanged} 个参数未变化")
        if changed > 0:
            print("✅✅✅ FFEM参数正在被训练！")
        else:
            print("❌❌❌ FFEM参数完全未变化，训练可能存在问题！")
        print("="*80 + "\n")
        self._initial_params.clear()


# ======================== 可视化回调 ========================
class AdvancedLossVisualizationCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics = {
            "train_loss": [], "train_steps": [],
            "eval_loss": [], "eval_steps": [],
            "learning_rate": [], "lr_steps": [],
            "grad_norm": [], "grad_steps": [],
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not _is_main_process():
            return
        step = state.global_step
        if "loss" in logs:
            self.metrics["train_loss"].append(float(logs["loss"]))
            self.metrics["train_steps"].append(step)
        if "learning_rate" in logs:
            self.metrics["learning_rate"].append(float(logs["learning_rate"]))
            self.metrics["lr_steps"].append(step)
        if "grad_norm" in logs:
            self.metrics["grad_norm"].append(float(logs["grad_norm"]))
            self.metrics["grad_steps"].append(step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics and _is_main_process():
            self.metrics["eval_loss"].append(float(metrics["eval_loss"]))
            self.metrics["eval_steps"].append(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if not _is_main_process():
            return
        self._plot()

    def _plot(self):
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except Exception:
            plt.style.use("default")
        plt.rcParams.update({
            "font.sans-serif": ["DejaVu Sans", "SimHei"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "font.size": 10,
        })

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
        C = {"train": "#2E86AB", "eval": "#A23B72", "lr": "#F18F01", "grad": "#06A77D"}

        ax1 = fig.add_subplot(gs[0, 0])
        if self.metrics["train_loss"]:
            ax1.plot(self.metrics["train_steps"], self.metrics["train_loss"],
                     color=C["train"], lw=2.5, alpha=0.8, label="Training Loss")
            if len(self.metrics["train_loss"]) > 10:
                z = np.polyfit(self.metrics["train_steps"], self.metrics["train_loss"], 3)
                ax1.plot(self.metrics["train_steps"],
                         np.poly1d(z)(self.metrics["train_steps"]),
                         "--", alpha=0.5, color="red", lw=2, label="Trend")
            ax1.set(title="Training Loss Curve", xlabel="Steps", ylabel="Loss")
            ax1.legend(); ax1.grid(True, alpha=0.3, ls="--"); ax1.set_facecolor("#F8F9FA")

        ax2 = fig.add_subplot(gs[0, 1])
        if self.metrics["eval_loss"]:
            ax2.plot(self.metrics["eval_steps"], self.metrics["eval_loss"],
                     color=C["eval"], lw=2.5, marker="o", ms=8,
                     mfc="white", mew=2, alpha=0.9, label="Validation Loss")
            mi = int(np.argmin(self.metrics["eval_loss"]))
            ax2.scatter([self.metrics["eval_steps"][mi]], [self.metrics["eval_loss"][mi]],
                        color="red", s=200, zorder=5, marker="*", edgecolors="black", lw=1.5)
            ax2.annotate(
                f'Best: {self.metrics["eval_loss"][mi]:.4f}',
                xy=(self.metrics["eval_steps"][mi], self.metrics["eval_loss"][mi]),
                xytext=(10, 10), textcoords="offset points", fontsize=10, color="red",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7))
            ax2.set(title="Validation Loss Curve", xlabel="Steps", ylabel="Loss")
            ax2.legend(); ax2.grid(True, alpha=0.3, ls="--"); ax2.set_facecolor("#F8F9FA")

        ax3 = fig.add_subplot(gs[1, :])
        if self.metrics["train_loss"]:
            ax3.plot(self.metrics["train_steps"], self.metrics["train_loss"],
                     color=C["train"], lw=2.5, alpha=0.7, label="Training Loss")
        if self.metrics["eval_loss"]:
            ax3.plot(self.metrics["eval_steps"], self.metrics["eval_loss"],
                     color=C["eval"], lw=3, marker="o", ms=8,
                     mfc="white", mew=2, alpha=0.9, label="Validation Loss")
        ax3.set(title="Training vs Validation Loss", xlabel="Steps", ylabel="Loss")
        ax3.legend(fontsize=12); ax3.grid(True, alpha=0.3, ls="--"); ax3.set_facecolor("#F8F9FA")

        ax4 = fig.add_subplot(gs[2, 0])
        if self.metrics["learning_rate"]:
            ax4.plot(self.metrics["lr_steps"], self.metrics["learning_rate"],
                     color=C["lr"], lw=2.5, label="Learning Rate")
            ax4.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax4.set(title="Learning Rate Schedule", xlabel="Steps", ylabel="LR")
            ax4.legend(); ax4.grid(True, alpha=0.3, ls="--"); ax4.set_facecolor("#F8F9FA")

        ax5 = fig.add_subplot(gs[2, 1])
        if self.metrics["grad_norm"]:
            ax5.plot(self.metrics["grad_steps"], self.metrics["grad_norm"],
                     color=C["grad"], lw=2.5, alpha=0.8, label="Gradient Norm")
            hi = [i for i, g in enumerate(self.metrics["grad_norm"]) if g > 100]
            if hi:
                ax5.scatter([self.metrics["grad_steps"][i] for i in hi],
                            [self.metrics["grad_norm"][i] for i in hi],
                            color="red", s=50, zorder=5, alpha=0.6, label="High (>100)")
            ax5.set(title="Gradient Norm", xlabel="Steps", ylabel="Norm")
            ax5.legend(); ax5.grid(True, alpha=0.3, ls="--"); ax5.set_facecolor("#F8F9FA")

        fig.suptitle("Training Metrics Dashboard - ThangkaVLM", fontsize=18, fontweight="bold")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output_dir / "training_metrics_dashboard.png", dpi=300,
                    bbox_inches="tight", facecolor="white")
        plt.savefig(self.output_dir / "training_metrics_dashboard.pdf",
                    bbox_inches="tight", facecolor="white")
        plt.close()

        with open(self.output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}\n✅ 训练指标已保存: {self.output_dir / 'training_metrics_dashboard.png'}")
        self._print_summary()
        print(f"{'='*70}\n")

    def _print_summary(self):
        m = self.metrics
        print(f"\n{'='*70}\n📈 训练摘要\n{'='*70}")
        if m["train_loss"]:
            print(f"  训练Loss: 初始={m['train_loss'][0]:.4f}, "
                  f"最终={m['train_loss'][-1]:.4f}, 最低={min(m['train_loss']):.4f}")
        if m["eval_loss"]:
            best_idx = m["eval_loss"].index(min(m["eval_loss"])) + 1
            print(f"  验证Loss: 初始={m['eval_loss'][0]:.4f}, "
                  f"最终={m['eval_loss'][-1]:.4f}, "
                  f"最优={min(m['eval_loss']):.4f} (第{best_idx}轮)")
        if m["learning_rate"]:
            print(f"  学习率: 最大={max(m['learning_rate']):.2e}, "
                  f"最终={m['learning_rate'][-1]:.2e}")
        if m["grad_norm"]:
            print(f"  梯度范数: 均值={np.mean(m['grad_norm']):.2f}, "
                  f"最大={max(m['grad_norm']):.2f}, 最小={min(m['grad_norm']):.2f}")
        print(f"{'='*70}")


# ======================== 自定义 Trainer ========================
class MaskedCausalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)

        # 在 forward 前把 region_labels 注入 merger，并从 inputs 中移除
        if "region_labels" in inputs:
            region_labels = inputs.pop("region_labels")
            merger = None
            # 按优先级查找merger
            for path in [
                lambda m: m.base_model.model.model.visual.merger,
                lambda m: m.base_model.model.visual.merger,
                lambda m: m.model.model.visual.merger,
                lambda m: m.model.visual.merger,
            ]:
                try:
                    merger = path(model)
                    if merger is not None:
                        break
                except AttributeError:
                    continue
            if merger is not None:
                merger.region_labels = region_labels

        outputs = model(**inputs)
        logits = outputs.logits

        if labels is None:
            loss = logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels.ne(-100)

        if mask.sum() == 0:
            loss = shift_logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        loss = F.cross_entropy(
            shift_logits[mask].float(), shift_labels[mask], reduction="mean"
        )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)

        if prediction_loss_only:
            return (loss, None, None)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
        return (loss, logits, inputs.get("labels"))


# ======================== 数据 Collator ========================
def data_collator_factory(tokenizer, image_pad_id: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {}
        for k in batch[0].keys():
            vals = [b[k] for b in batch]
            if torch.is_tensor(vals[0]):
                if k == "region_labels":
                    max_len = max(v.shape[0] for v in vals)
                    padded = torch.zeros(len(vals), max_len, dtype=torch.long)
                    for i, v in enumerate(vals):
                        padded[i, :v.shape[0]] = v
                    out[k] = padded
                else:
                    out[k] = torch.stack(vals, 0)
            else:
                out[k] = vals
        return out
    return collate


# ======================== 模型保存工具 ========================
def _save_trainable_bin(model, output_dir, filename="pytorch_model_ffem.bin"):
    if not _is_main_process():
        return
    sd = model.state_dict()
    delta = {k: v.detach().cpu() for k, v in sd.items()
             if _is_lora_name(k) or _is_ffem_name(k)}
    if not delta:
        print("⚠️ 未找到LoRA/FFEM权重，跳过保存")
        return
    path = Path(output_dir) / filename
    torch.save(delta, path)
    print(f"✅ 已保存LoRA+FFEM权重 → {path} (共{len(delta)}个参数)")


def _merge_ffem_into_adapter_safetensors(model, output_dir):
    if not _is_main_process():
        return
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as e:
        print(f"⚠️ safetensors未安装，跳过合并: {e}")
        return
    adapter_path = Path(output_dir) / "adapter_model.safetensors"
    if not adapter_path.exists():
        print(f"⚠️ 未找到{adapter_path}，跳过合并")
        return
    sd = model.state_dict()
    ffem_sd = {k: v.detach().cpu() for k, v in sd.items() if _is_ffem_name(k)}
    if not ffem_sd:
        print("⚠️ 未发现FFEM权重，跳过合并")
        return
    merged = dict(load_file(str(adapter_path)))
    merged.update(ffem_sd)
    save_file(merged, str(adapter_path))
    print(f"✅ FFEM权重已合并到 {adapter_path} (新增{len(ffem_sd)}个tensor)")


def _save_pure_ffem_weights(model, output_dir, filename="pytorch_model_pure_ffem.bin"):
    if not _is_main_process():
        return
    sd = model.state_dict()
    ffem_sd = {k: v.detach().cpu() for k, v in sd.items() if _is_ffem_name(k)}
    if not ffem_sd:
        print("⚠️ 未找到FFEM权重，跳过保存")
        return
    path = Path(output_dir) / filename
    torch.save(ffem_sd, path)
    print(f"✅ 已保存纯FFEM权重 → {path} (共{len(ffem_sd)}个参数)")


# ======================== 梯度检查点管理 ========================
def _enable_non_reentrant_checkpointing(model):
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if _is_main_process():
            print("✅ 梯度检查点：启用非重入式模式")
        return True
    except (TypeError, AttributeError):
        model.gradient_checkpointing_enable()
        if _is_main_process():
            print("⚠️ 梯度检查点：退化为默认重入式模式")
        return False


def _disable_visual_gradient_checkpointing(model):
    visual = None
    candidates = [
        lambda m: m.base_model.model.model.visual,
        lambda m: m.base_model.model.visual,
        lambda m: m.model.model.visual,
        lambda m: m.model.visual,
    ]
    for getter in candidates:
        try:
            visual = getter(model)
            if visual is not None:
                break
        except AttributeError:
            continue
    if visual is None:
        if _is_main_process():
            print("⚠️ 未找到视觉编码器，跳过关闭梯度检查点")
        return

    count = sum(1 for m in visual.modules() if hasattr(m, "gradient_checkpointing"))
    for m in visual.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = False
    if _is_main_process():
        print(f"✅ 已关闭{count}个视觉模块的梯度检查点")

    remaining = sum(1 for m in visual.modules()
                    if getattr(m, "gradient_checkpointing", False))
    if remaining > 0:
        for m in visual.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = False
        if _is_main_process():
            print("✅ 强制关闭完成")
    else:
        if _is_main_process():
            print("✅ 验证通过：所有视觉模块梯度检查点已关闭")


def _setup_device_and_rank(args):
    env_lr = os.environ.get("LOCAL_RANK")
    if env_lr is not None:
        try:
            args.local_rank = int(env_lr)
        except ValueError:
            args.local_rank = -1
    if torch.cuda.is_available():
        if args.local_rank >= 0:
            torch.cuda.set_device(args.local_rank)
            if _is_main_process():
                print(f"✅ 分布式训练：LOCAL_RANK={args.local_rank}, "
                      f"WORLD_SIZE={os.environ.get('WORLD_SIZE', 1)}")
        else:
            if _is_main_process():
                print("✅ 单机训练模式（CUDA可用）")
    else:
        if _is_main_process():
            print("⚠️ CUDA不可用，使用CPU训练（速度极慢）")


# ======================== 主函数 ========================
def main():
    args = parse_args()

    os.environ.setdefault("TORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    _setup_device_and_rank(args)

    # 数据集路径
    data_dir = Path(args.data_dir).resolve()
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if "train" in str(data_dir) and not train_dir.exists():
        train_dir = data_dir
        val_dir = Path(str(data_dir).replace("/train", "/val"))
    elif "val" in str(data_dir) and not val_dir.exists():
        val_dir = data_dir
        train_dir = Path(str(data_dir).replace("/val", "/train"))

    for d, label in [(train_dir, "训练集"), (val_dir, "验证集")]:
        if not d.exists():
            raise RuntimeError(f"❌ {label}路径不存在: {d}")
        if _is_main_process():
            print(f"✅ {label}路径：{d}")

    if _is_main_process():
        print(f"\n{'='*70}\n📋 训练配置\n{'='*70}")
        print(f"模型路径：{args.model_path}")
        print(f"输出目录：{args.output_dir}")
        print(f"微调模式：{args.tune}")
        print(f"学习率：LoRA={args.learning_rate:.2e} | FFEM={args.learning_rate*2:.2e}")
        print(f"批次配置：{args.batch_size} × {args.gradient_accumulation} "
              f"= {args.batch_size*args.gradient_accumulation}")
        print(f"训练轮数：{args.epochs} | 早停耐心值：{args.early_stop_patience}")
        print(f"精度配置：bf16={args.bf16}, fp16={args.fp16}")
        print(f"{'='*70}\n")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if _is_main_process():
        print("🔄 加载tokenizer和processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)

    # 数据集验证
    if _is_main_process():
        print("🔄 验证数据集加载...")
    ds_check = _td.ThangkaDataset(
        data_dir=str(train_dir), processor=processor, tokenizer=tokenizer,
        split="train", task_mode=args.task_mode, use_bbox=args.use_bbox,
        augment_qa=args.augment_qa, user_max_tokens=args.user_max_tokens,
        assistant_max_tokens=args.assistant_max_tokens,
        max_seq_len=args.max_seq_len, image_max_side=args.image_max_side,
    )
    if len(ds_check) == 0:
        raise RuntimeError(f"❌ 训练集为空：{train_dir}")
    sample = ds_check[0]
    for key in ["image_grid_thw", "pixel_values", "region_labels"]:
        if key not in sample:
            raise RuntimeError(f"❌ 数据集未返回{key}字段")
    if _is_main_process():
        print(f"\n✅ 数据集维度检查:")
        print(f"   pixel_values: {sample['pixel_values'].shape}")
        print(f"   image_grid_thw: {sample['image_grid_thw']}")
        print(f"   region_labels: {sample['region_labels'].shape}, "
              f"非零数: {sample['region_labels'].sum().item()}")
        if sample['region_labels'].sum() == 0:
            print("⚠️ 警告：region_labels全为0，已自动修复！")

    # 加载模型
    if _is_main_process():
        print("🔄 加载模型...")
    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = args.fp16 and torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    model = Qwen3VLWithFFEM.from_pretrained(
        args.model_path, local_files_only=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )

    if _is_main_process():
        print(f"🔄 应用微调模式：{args.tune}")
    model = apply_tuning_mode(args, model)

    # 顺序固定：先整体开启，再关闭视觉侧
    _enable_non_reentrant_checkpointing(model)
    _disable_visual_gradient_checkpointing(model)
    model.config.use_cache = False

    try:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "base_model"):
            model.base_model.enable_input_require_grads()
    except Exception as e:
        if _is_main_process():
            print(f"⚠️ enable_input_require_grads失败（可忽略）: {e}")

    # 加载数据集
    if _is_main_process():
        print("🔄 加载训练/验证数据集...")
    train_dataset = _td.ThangkaDataset(
        data_dir=str(train_dir), processor=processor, tokenizer=tokenizer,
        split="train", task_mode=args.task_mode, use_bbox=args.use_bbox,
        augment_qa=args.augment_qa, user_max_tokens=args.user_max_tokens,
        assistant_max_tokens=args.assistant_max_tokens,
        max_seq_len=args.max_seq_len, image_max_side=args.image_max_side,
    )
    eval_dataset = _td.ThangkaDataset(
        data_dir=str(val_dir), processor=processor, tokenizer=tokenizer,
        split="val", task_mode=args.task_mode, use_bbox=args.use_bbox,
        augment_qa=False, user_max_tokens=args.user_max_tokens,
        assistant_max_tokens=args.assistant_max_tokens,
        max_seq_len=args.max_seq_len, image_max_side=args.image_max_side,
    )
    if _is_main_process():
        print(f"✅ 加载了 {len(train_dataset)} 条 train 数据 (data_dir={train_dir})")
        print(f"✅ 加载了 {len(eval_dataset)} 条 val 数据 (data_dir={val_dir})")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=False,
        logging_steps=args.logging_steps,
        save_steps=1,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=args.weight_decay,
        eval_accumulation_steps=4,
        prediction_loss_only=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
    )

    param_groups = get_parameter_groups(args, model)
    optimizer = optim.AdamW(
        param_groups, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8, weight_decay=args.weight_decay,
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    num_training_steps = len(train_dataset) * args.epochs // (
        args.batch_size * args.gradient_accumulation * world_size)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    class DisableVisualCheckpointingCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if model is None or not _is_main_process():
                return
            try:
                visual = model.base_model.model.model.visual
                count = sum(1 for m in visual.modules()
                            if getattr(m, "gradient_checkpointing", False))
                if count > 0:
                    for m in visual.modules():
                        if hasattr(m, "gradient_checkpointing"):
                            m.gradient_checkpointing = False
                    print(f"✅ [训练开始] 再次关闭{count}个视觉模块的梯度检查点")
            except AttributeError:
                pass

    trainer = MaskedCausalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator_factory(
            tokenizer, image_pad_id=tokenizer.pad_token_id or 0),
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[
            DisableVisualCheckpointingCallback(),
            EarlyStoppingCallback(patience=args.early_stop_patience),
            FFEMGradientCheckCallback(check_after_steps=10),
            AdvancedLossVisualizationCallback(output_dir=args.output_dir),
        ],
    )

    if _is_main_process():
        print("\n🚀 开始训练...")
        print(f"   训练集大小：{len(train_dataset)}")
        print(f"   验证集大小：{len(eval_dataset)}")
        print(f"   总训练步数：{num_training_steps}")
        print(f"   预热步数：{num_warmup_steps}")
        print("="*70 + "\n")

    trainer.train()

    # 保存
    save_ok = False
    try:
        save_ok = trainer.is_world_process_zero()
    except Exception:
        save_ok = _is_main_process()

    if save_ok:
        print("\n🔄 保存模型...")
        try:
            tokenizer.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print("✅ 已保存tokenizer和processor")
        except Exception as e:
            print(f"⚠️ 保存tokenizer/processor失败: {e}")
        try:
            if hasattr(trainer.model, "save_pretrained"):
                trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
                print(f"✅ 已保存adapter模型到：{args.output_dir}")
        except Exception as e:
            print(f"⚠️ 保存adapter模型失败: {e}")

        _save_trainable_bin(trainer.model, args.output_dir)
        _merge_ffem_into_adapter_safetensors(trainer.model, args.output_dir)
        _save_pure_ffem_weights(trainer.model, args.output_dir)

        print(f"\n{'='*70}\n🎉 训练完成！\n📁 模型输出目录：{args.output_dir}")
        if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
            print(f"🏆 最优验证集Loss：{trainer.state.best_metric:.4f}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败：{e}")
        raise