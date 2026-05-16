# ThangkaVLM - 唐卡视觉语言模型

基于Qwen3-VL + FFEM的唐卡专用视觉语言模型，支持元素检测、属性识别和问答任务。

## 🌟 特性

- ✅ **多任务支持**: 问答、元素检测、属性识别
- ✅ **FFEM增强**: 三阶段特征增强模块 (MSFR + SRE + SSAM)
- ✅ **自动数据增强**: 根据标注自动生成更多问答对
- ✅ **灵活训练策略**: 支持部分冻结、全参数微调等
- ✅ **完整工具链**: 数据组织、验证、训练、推理全流程

## 📦 快速开始

### 一键运行

```bash
# 1. 赋予执行权限
chmod +x quick_start.sh

# 2. 运行脚本
./quick_start.sh
```

按照提示完成：数据组织 → 验证 → 训练

### 手动运行

#### 1. 安装依赖

```bash
pip install torch transformers accelerate deepspeed Pillow tensorboard
```

#### 2. 组织数据

```bash
python convert_thangka_data.py \
    --mode organize \
    --source_dir ./raw_data \
    --target_dir ./thangka_data
```

#### 3. 验证数据

```bash
python convert_thangka_data.py \
    --mode validate \
    --data_dir ./thangka_data
```

#### 4. 开始训练

```bash
# 推荐配置
python train_thangka_custom.py \
    --model_path ./ThangkaVLM \
    --data_dir ./thangka_data \
    --output_dir ./output \
    --task_mode mixed \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --freeze_vision
```

#### 5. 测试模型

```bash
python inference.py \
    --model_path ./output \
    --mode chat \
    --image test.jpg
```

## 📁 项目结构

```
ThangkaVLM/
├── ffem.py                      # FFEM模块实现
├── modeling_qwen3_vl_ffem.py    # 自定义模型类
├── model_registration.py        # 模型注册
├── thangka_dataset.py          # 唐卡数据集加载器 ⭐
├── train_thangka_custom.py     # 训练脚本 ⭐
├── convert_thangka_data.py     # 数据转换工具 ⭐
├── inference.py                 # 推理脚本
├── quick_start.sh              # 一键启动脚本 ⭐
├── config.json                  # 模型配置
├── *.safetensors               # 模型权重
└── README.md                    # 本文件
```

⭐ = 为你的数据集新创建的文件

## 📊 数据格式

你的JSON标注格式：

```json
{
    "instances": [
        {
            "image_id": "003.jpg",
            "is_mirrored": true,
            "elements": [
                {
                    "label": "YellowJambhala",
                    "name_zh": "黄财神",
                    "role": "main_deity",
                    "bbox": [232, 223, 1398, 1631]
                }
            ],
            "qa_pairs": [
                {
                    "question": "图片中心的人物是谁？",
                    "answer": "黄财神",
                    "question_type": "identification"
                }
            ]
        }
    ]
}
```

数据目录结构：

```
thangka_data/
├── images/
│   ├── 003.jpg
│   └── ...
└── annotations/
    ├── 003.json
    └── ...
```

## 🎯 训练策略

### 策略对比

| 策略 | 显存需求 | 训练时间 | 效果 | 适用场景 |
|------|---------|---------|------|---------|
| 只训练FFEM | ~15GB | 快 | 良好 | 快速验证 |
| FFEM+语言模型 | ~25GB | 中等 | 很好 | **推荐** |
| 全参数微调 | ~40GB | 慢 | 最佳 | 追求极致 |

### 推荐配置

```bash
python train_thangka_custom.py \
    --model_path ./ThangkaVLM \
    --data_dir ./thangka_data \
    --output_dir ./output \
    --task_mode mixed \          # 混合任务
    --epochs 3 \                 # 3轮
    --batch_size 2 \             # batch size 2
    --gradient_accumulation 8 \  # 梯度累积8步
    --learning_rate 2e-5 \       # 学习率2e-5
    --freeze_vision              # 冻结视觉编码器
```

## 🔧 训练参数说明

| 参数 | 说明 | 默认值 | 建议范围 |
|------|------|--------|---------|
| `--task_mode` | 任务模式 | mixed | qa/detection/mixed |
| `--epochs` | 训练轮数 | 5 | 3-10 |
| `--batch_size` | Batch大小 | 2 | 1-4 |
| `--gradient_accumulation` | 梯度累积 | 8 | 4-16 |
| `--learning_rate` | 学习率 | 2e-5 | 1e-5~5e-5 |
| `--freeze_vision` | 冻结视觉 | False | 建议True |
| `--freeze_llm` | 冻结语言 | False | 快速验证时True |

## 📈 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir ./output/logs

# 访问 http://localhost:6006
```

查看：
- 训练/验证损失
- 学习率变化
- 梯度分布
- 参数更新

## 🎨 推理示例

### 单图推理

```bash
python inference.py \
    --model_path ./output \
    --mode single \
    --image test.jpg \
    --prompt "描述这幅唐卡"
```

### 交互对话

```bash
python inference.py \
    --model_path ./output \
    --mode chat \
    --image test.jpg
```

示例对话：
```
👤 You: 主尊是谁？
🤖 ThangkaVLM: 黄财神

👤 You: 主尊左手持什么？
🤖 ThangkaVLM: 吐宝鼠

👤 You: 描述主尊的姿态
🤖 ThangkaVLM: 主尊黄财神坐在莲花座上，左手持吐宝鼠，右手持布拉噶如意宝...
```

### 批量测试

```bash
python inference.py \
    --model_path ./output \
    --mode batch \
    --test_file test_data.json \
    --output_file results.json
```

## 🚀 性能优化

### 显存优化

1. **减小batch size**
   ```bash
   --batch_size 1 --gradient_accumulation 16
   ```

2. **使用DeepSpeed**
   ```bash
   deepspeed --num_gpus=1 train_thangka_custom.py \
       --deepspeed ds_z2_config.json \
       ...
   ```

3. **冻结参数**
   ```bash
   --freeze_vision --freeze_llm
   ```

### 速度优化

1. **多GPU训练**
   ```bash
   deepspeed --num_gpus=2 train_thangka_custom.py ...
   ```

2. **混合精度**
   - 自动启用FP16

3. **增加workers**
   - 数据加载默认使用4个worker

## 📚 核心功能

### 1. 自动数据增强

根据元素标注自动生成问答：

- ✅ 存在性问题: "图中是否有黄财神？"
- ✅ 识别问题: "主尊是谁？"
- ✅ 属性问题: "主尊左手持什么？"
- ✅ 位置问题: "莲花座在哪里？"

### 2. 多任务学习

- **问答任务**: 回答关于唐卡的问题
- **检测任务**: 识别和定位元素
- **混合任务**: 结合以上所有任务

### 3. FFEM增强

- **MSFR**: 多尺度特征表示
- **SRE**: 显著区域增强
- **SSAM**: 空间语义对齐（可选region labels）

## ⚠️ 常见问题

### 1. CUDA内存溢出

```bash
# 减小batch size
--batch_size 1 --gradient_accumulation 16

# 或使用DeepSpeed
--deepspeed ds_z2_config.json
```

### 2. 数据加载失败

```bash
# 验证数据
python convert_thangka_data.py --mode validate --data_dir ./thangka_data

# 预览数据
python convert_thangka_data.py --mode preview --data_dir ./thangka_data
```

### 3. 训练loss不下降

- 检查学习率（试试1e-5）
- 不要冻结太多参数
- 检查数据质量和数量

### 4. 生成质量差

- 增加训练数据（建议100+）
- 训练更多epoch（5-10）
- 调整temperature（0.3-1.0）
- 不要冻结语言模型

## 🎓 最佳实践

1. ✅ **先验证数据**: 确保格式正确
2. ✅ **小规模测试**: 先用少量数据验证流程
3. ✅ **监控训练**: 使用TensorBoard
4. ✅ **保存多个checkpoint**: 方便回退
5. ✅ **逐步解冻**: 先冻结多，再逐步解冻

## 📖 详细文档

- [完整训练指南](完整训练指南.md) - 详细的训练流程和参数说明
- [数据格式说明](数据格式说明.md) - JSON格式详解
- [常见问题解答](FAQ.md) - 疑难问题解决

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 License

MIT License

---

**开始你的唐卡VLM训练之旅！** 🎨✨

如有问题，请查看日志或在GitHub提Issue。