# ThangkaVLM - Thangka Vision-Language Model

Official implementation of T-HVA and FFEM for reliable visual question answering on structured cultural heritage images.

> ⚠️ **Important Notice**
> The codes, datasets, and evaluation scripts in this repository are directly related to the manuscript currently submitted to **The Visual Computer**.
> Readers and researchers using this repository are encouraged to cite the corresponding manuscript once published.

---

## 🌟 Features

* ✅ **Multi-task Support**: Visual question answering, element detection, and attribute recognition
* ✅ **FFEM Enhancement**: Three-stage feature enhancement module (MSFR + SRE + SSAM)
* ✅ **Automatic Data Augmentation**: Automatically generates additional QA pairs from annotations
* ✅ **Flexible Training Strategies**: Supports partial freezing and full fine-tuning
* ✅ **Complete Toolchain**: End-to-end pipeline including data organization, validation, training, and inference

---

## 📦 Quick Start

### One-Click Run

```bash
# 1. Grant execution permission
chmod +x quick_start.sh

# 2. Run the script
./quick_start.sh
```

Follow the prompts to complete:

```text
Data organization → Validation → Training
```

---

## 🔧 Manual Setup

### 1. Install Dependencies

```bash
pip install torch transformers accelerate deepspeed Pillow tensorboard
```

### 2. Organize Data

```bash
python convert_thangka_data.py \
    --mode organize \
    --source_dir ./raw_data \
    --target_dir ./thangka_data
```

### 3. Validate Data

```bash
python convert_thangka_data.py \
    --mode validate \
    --data_dir ./thangka_data
```

### 4. Start Training

```bash
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

### 5. Test the Model

```bash
python inference.py \
    --model_path ./output \
    --mode chat \
    --image test.jpg
```

---

## 📁 Project Structure

```text
ThangkaVLM/
├── ffem.py                      # FFEM module implementation
├── modeling_qwen3_vl_ffem.py   # Customized model class
├── model_registration.py       # Model registration
├── thangka_dataset.py          # Dataset loader
├── train_thangka_custom.py     # Training script
├── convert_thangka_data.py     # Data conversion tool
├── inference.py                # Inference script
├── quick_start.sh              # Quick startup script
├── config.json                 # Model configuration
├── requirements.txt            # Dependency list
├── datasets/                   # Dataset splits and annotations
└── README.md                   # Documentation
```

---

## 📊 Data Format

Example JSON annotation format:

```json
{
    "instances": [
        {
            "image_id": "003.jpg",
            "is_mirrored": true,
            "elements": [
                {
                    "label": "YellowJambhala",
                    "name_zh": "Yellow Jambhala",
                    "role": "main_deity",
                    "bbox": [232, 223, 1398, 1631]
                }
            ],
            "qa_pairs": [
                {
                    "question": "Who is the figure at the center of the image?",
                    "answer": "Yellow Jambhala",
                    "question_type": "identification"
                }
            ]
        }
    ]
}
```

Dataset directory structure:

```text
thangka_data/
├── images/
│   ├── 003.jpg
│   └── ...
└── annotations/
    ├── 003.json
    └── ...
```

---

## 🎯 Training Strategies

| Strategy         | GPU Memory | Training Time | Performance | Application         |
| ---------------- | ---------- | ------------- | ----------- | ------------------- |
| FFEM Only        | ~15GB      | Fast          | Good        | Quick verification  |
| FFEM + LLM       | ~25GB      | Medium        | Very Good   | Recommended         |
| Full Fine-tuning | ~40GB      | Slow          | Best        | Maximum performance |

---

## 📈 Training Monitoring

```bash
tensorboard --logdir ./output/logs
```

Visit:

```text
http://localhost:6006
```

Monitor:

* Training/validation loss
* Learning rate schedule
* Gradient distribution
* Parameter updates

---

## 🎨 Inference Examples

### Single Image Inference

```bash
python inference.py \
    --model_path ./output \
    --mode single \
    --image test.jpg \
    --prompt "Describe this Thangka image"
```

### Interactive Chat

```bash
python inference.py \
    --model_path ./output \
    --mode chat \
    --image test.jpg
```

---

## 🚀 Performance Optimization

### GPU Memory Optimization

```bash
--batch_size 1 --gradient_accumulation 16
```

### DeepSpeed Training

```bash
deepspeed --num_gpus=1 train_thangka_custom.py \
    --deepspeed ds_z2_config.json
```

### Multi-GPU Training

```bash
deepspeed --num_gpus=2 train_thangka_custom.py
```

---

## 📚 Core Functions

### Automatic Data Augmentation

Automatically generates QA pairs from element annotations:

* Existence questions
* Identification questions
* Attribute questions
* Location questions

### Multi-task Learning

* Visual Question Answering
* Element Detection
* Mixed-task Training

### FFEM Enhancement

* MSFR: Multi-scale Feature Reconstruction
* SRE: Salient Region Enhancement
* SSAM: Spatial Semantic Alignment Mixing

---

## ⚠️ Notes

* Large pretrained model weights are not included in this repository.
* Some image files may not be publicly released due to copyright restrictions.
* Dataset annotations and evaluation scripts are provided for reproducibility.

---

## 📖 Citation

If you find this repository useful, please consider citing the related manuscript submitted to **The Visual Computer**.

```bibtex
@article{thangkavlm2025,
  title={Structure-Aware Fine-Grained Visual Enhancement for Reliable Visual Question Answering on Structured Cultural Heritage Images},
  author={Anonymous},
  journal={The Visual Computer},
  year={2025}
}
```

---

## 🤝 Contribution

Issues and Pull Requests are welcome.

---

## 📄 License

MIT License

---

**Start your ThangkaVLM journey today!** 🎨✨
