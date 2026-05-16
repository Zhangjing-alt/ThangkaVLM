# ThangkaVLM - Thangka Vision-Language Model

Official implementation of T-HVA and FFEM for reliable visual question answering on structured cultural heritage images.

> ⚠️ **Important Notice**
> The codes, datasets, and evaluation scripts in this repository are directly related to the manuscript currently submitted to **The Visual Computer**.
> Readers and researchers using this repository are encouraged to cite the corresponding manuscript once published.

---

## 🌟 Features

* ✅ **Multi-task Support**: Visual question answering, element detection, and attribute recognition
* ✅ **FFEM Enhancement**: Three-stage feature enhancement module (MSFR + SRE + SSAM)
* ✅ **Flexible Training Strategies**: Supports partial freezing and full fine-tuning
* ✅ **Complete Toolchain**: End-to-end pipeline including data organization, validation, training, and inference

---

## 📦 Quick Start

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

### 2. Start Training

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
---

## 📁 Project Structure

```text
ThangkaVLM/
├── ffem.py                      # FFEM module implementation
├── modeling_qwen3_vl_ffem.py   # Customized model class
├── model_registration.py       # Model registration
├── thangka_dataset.py          # Dataset loader
├── train_thangka_custom.py     # Training script
├── config.json                 # Model configuration
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
