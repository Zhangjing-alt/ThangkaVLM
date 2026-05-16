````markdown
# Structure-Aware Fine-Grained Visual Enhancement for Reliable Visual Question Answering on Structured Cultural Heritage Images

This repository is the official implementation of the manuscript submitted to **The Visual Computer**.

---

## Overview

We propose:

- **T-HVA**: Thangka Hierarchical Visual Alignment
- **FFEM**: Fine-Grained Feature Extraction Module

for reliable visual question answering on highly structured cultural heritage images.

The proposed method improves fine-grained visual grounding and reduces multimodal hallucinations in structured visual scenarios.

---

## Dataset

### Thangka-VQA

The Thangka-VQA dataset contains:

- 1,028 Thangka images
- 10,024 question-answer pairs
- Standard questions
- Adversarial questions

Dataset structure:

```text
datasets/
    train.json
    val.json
    test.json
````

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Inference

```bash
python inference.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Repository Structure

```text
ThangkaVLM/
│
├── README.md
├── requirements.txt
├── train.py
├── inference.py
├── evaluate.py
├── datasets/
├── checkpoints/
└── scripts/
```

---

## Citation

If you find this work useful, please cite our paper.

```bibtex
@article{thangkavlm2025,
  title={Structure-Aware Fine-Grained Visual Enhancement for Reliable Visual Question Answering on Structured Cultural Heritage Images},
  author={Your Name},
  journal={The Visual Computer},
  year={2025}
}
```

---

## License

This project is released for academic research purposes only.

```
```
