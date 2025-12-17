![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Status](https://img.shields.io/badge/status-Research%20Prototype-yellow.svg)
![Adversarial](https://img.shields.io/badge/Adversarial-FGSM%20%7C%20PGD-red.svg)
![Entropy](https://img.shields.io/badge/Entropy-Reliability%20Monitoring-purple.svg)

# Robust Entropy-Based CNN Reliability Framework

> **Entropy-aware reliability monitoring and selective inference for convolutional neural networks under adversarial attacks**

---

# 📘 Abstract

> **Abstract —**
> Deep neural networks achieve high predictive accuracy but remain vulnerable to adversarial perturbations and silent failures. Recent work introduced entropy-based, non-invasive reliability monitoring as a means of estimating prediction uncertainty without modifying model architectures.
>
> In this work, we extend entropy-based reliability monitoring beyond a single baseline convolutional neural network by conducting a systematic, architecture-aware study across multiple CNN designs. We introduce a hybrid entropy-aware CNN that integrates transfer learning, grouped and depthwise convolutions, and attention mechanisms to improve robustness and reliability under adversarial conditions.
>
> We evaluate clean performance, adversarial robustness under FGSM and PGD attacks, entropy–correctness correlation, selective classification trade-offs, and inference latency. Our results confirm that entropy consistently reflects predictive uncertainty, while architectural enhancements significantly improve entropy separability and selective reliability. However, entropy alone does not prevent adversarial failure, highlighting the necessity of combining monitoring with robustness-aware training.
>
> This study transforms entropy from a diagnostic signal into an actionable reliability mechanism and provides practical guidance for deploying entropy-based monitoring in safety-critical systems.
---
## 🔍 Overview

This repository extends the paper **“Entropy-Based Non-Invasive Reliability Monitoring of Convolutional Neural Networks”** by transforming entropy from a *passive diagnostic signal* into an **actionable reliability mechanism**.

We evaluate entropy behavior across **multiple CNN architectures**, introduce a **hybrid CNN with transfer learning**, and operationalize entropy via **selective classification** under **strong adversarial attacks (FGSM & PGD)**.

---

## 📄 Paper Alignment

**Original Paper Contribution**

* Shows that internal activation entropy correlates with prediction reliability
* Uses entropy as a *monitoring signal*
* Evaluated on a single CNN architecture
* Focused primarily on FGSM attacks

**This Project Extends**

* Architecture diversity (Simple CNN, MobileNet, Hybrid CNN)
* Stronger adversarial evaluation (PGD)
* Entropy-driven **decision control**
* Practical deployment metrics (coverage, latency)

---

## 🚀 Key Contributions

* **Architecture-Aware Entropy Analysis**

  * Simple CNN
  * MobileNet
  * Hybrid CNN (depthwise + grouped + attention)

* **Selective Classification**

  * Entropy-based rejection
  * Accuracy–coverage trade-off
  * Reliability-aware inference

* **Hybrid CNN (Phase 2)**

  * MobileNetV2 transfer learning
  * Grouped & depthwise convolutions
  * Squeeze-and-Excitation attention
  * Optional 3D path for volumetric data

* **Robustness Evaluation**

  * FGSM (single-step)
  * PGD (iterative, strong attack)

---

## 🧪 Experimental Phases

### Phase 1 — Baseline Validation

* High clean accuracy
* Severe accuracy collapse under FGSM/PGD
* Entropy correlates with failure but does not prevent it

### Phase 2 — Hybrid Architecture

* Higher clean accuracy
* Improved entropy separation
* Better selective reliability
* Partial robustness gains

### Phase 3 — Comparative Evaluation

* Phase 1 vs Phase 2
* Accuracy, robustness, coverage, latency

---

## 📊 Results Snapshot

| Metric             | Phase 1    | Phase 2      |
| ------------------ | ---------- | ------------ |
| Clean Accuracy     | ~73.7%     | ~85.5%       |
| FGSM Accuracy      | ~1.5%      | ~10%         |
| PGD Accuracy       | ~0%        | ~10%         |
| Selective Accuracy | 73.6% @97% | 85.4% @98.7% |

> **Conclusion:**
> Entropy is a reliable uncertainty signal but **not a defense by itself**.

---

## 📁 Project Structure

```
RobustEntropyCNN/
├── models/
│   ├── base_models/
│   │   ├── simple_cnn.py
│   │   ├── mobilenet_baseline.py
│   │   └── hybrid_cnn.py
│   └── layers/
│       ├── depthwise_blocks.py
│       ├── grouped_conv.py
│       └── attention_se.py
├── attacks/
│   ├── fgsm.py
│   └── pgd.py
├── evaluation/
│   ├── entropy.py
│   ├── selective_classification.py
│   └── metrics.py
├── experiments/
│   ├── train.py
│   ├── run_attack_pipeline.py
│   └── run_visualization_pipeline.py
└── README.md
```

---

## 🧠 Key Takeaways

* Entropy consistently reflects prediction uncertainty
* Stronger architectures improve entropy separability
* Selective classification converts uncertainty into reliability control
* Robustness–accuracy trade-off remains unresolved without training-time defenses

---

## 🔮 Future Work

* Entropy-guided adversarial training
* Adaptive, learned entropy thresholds
* Multi-layer entropy fusion
* Real-world datasets (autonomous driving, medical imaging)

---

## 📌 Final Statement

> This project validates and **extends** entropy-based reliability monitoring from a conceptual analysis to a **deployable, architecture-aware reliability framework**, without contradicting the original paper’s claims.
---

# 🧪 Reproducibility Checklist

### Environment

* Python ≥ 3.9
* TensorFlow 2.15
* CUDA optional (CPU supported)

### Dataset

* CIFAR-10 (official Keras loader)
* Fixed train/validation/test split
* Seeded randomness for reproducibility

### Models

* Simple CNN (baseline)
* MobileNet-based CNN
* Hybrid CNN (transfer learning + grouped + attention)

### Attacks

* FGSM (single-step)
* PGD (iterative)

### Metrics

* Clean accuracy
* Adversarial accuracy
* Entropy distributions
* Selective accuracy vs coverage
* Inference latency

### Reproducibility Guarantees

* Deterministic seeds
* Fixed preprocessing
* Explicit entropy thresholding
* Saved trained models

---

# 🧾 Paper vs Project Comparison Table

| Aspect                 | Original Paper                   | This Project                             |
| ---------------------- | -------------------------------- | ---------------------------------------- |
| Goal                   | Entropy as reliability indicator | Entropy as reliability **mechanism**     |
| Architecture           | Single CNN                       | Multiple CNNs + Hybrid CNN               |
| Entropy Usage          | Monitoring only                  | Monitoring + selective inference         |
| Adversarial Attacks    | FGSM                             | FGSM + PGD                               |
| Decision Control       | None                             | Entropy-based rejection                  |
| Transfer Learning      | ❌                                | ✅                                        |
| Attention Mechanisms   | ❌                                | ✅ (SE blocks)                            |
| Multi-Domain Readiness | ❌                                | ✅ (2D / 3D path)                         |
| Deployment Metrics     | Limited                          | Accuracy, coverage, latency              |
| Conclusion             | Entropy correlates with failure  | Entropy is useful but insufficient alone |

---
##  Author
**Omar Khalil**

MSc Researcher @ AASTMT

Connect on [LinkedIn](https://www.linkedin.com/in/omar-khalil-10om01) or check more projects!

---

## License

This project is for **Research purposes**.
---
