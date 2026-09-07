# model_registration.py
# -*- coding: utf-8 -*-

"""
幂等注册 Qwen3VLWithFFEM。
Transformers 新版已经自带 qwen3_vl -> Qwen3VLForConditionalGeneration 的映射，
我们这里“尽量覆盖”，如果不允许覆盖就静默跳过（避免 DeepSpeed 多进程重复注册报错）。
"""

from transformers import AutoConfig

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from modeling_qwen3_vl_ffem import Qwen3VLWithFFEM


def _register_config_idempotent():
    try:
        AutoConfig.register("qwen3_vl", Qwen3VLConfig)
    except Exception:
        # 已注册就忽略
        pass


def _register_model_idempotent():
    if AutoModelForImageTextToText is None:
        print("⚠️ AutoModelForImageTextToText 不可用，跳过 Auto 映射注册。")
        return

    # Transformers 已经有默认映射了，这里尝试覆盖；不允许覆盖就忽略
    try:
        AutoModelForImageTextToText.register(Qwen3VLConfig, Qwen3VLWithFFEM, exist_ok=True)
        print("✅ 已注册/覆盖 AutoModelForImageTextToText: Qwen3VLConfig -> Qwen3VLWithFFEM")
        return
    except TypeError:
        # 某些版本没有 exist_ok 参数
        pass
    except ValueError:
        # 已被占用且不允许覆盖，忽略
        print("ℹ️ Auto 映射已存在且不允许覆盖，跳过注册（不影响直接用 Qwen3VLWithFFEM.from_pretrained）。")
        return

    # 兜底：无 exist_ok 参数时捕获 ValueError
    try:
        AutoModelForImageTextToText.register(Qwen3VLConfig, Qwen3VLWithFFEM)
        print("✅ 已注册 AutoModelForImageTextToText: Qwen3VLConfig -> Qwen3VLWithFFEM")
    except ValueError:
        print("ℹ️ Auto 映射已存在，跳过注册。")


_register_config_idempotent()
_register_model_idempotent()
