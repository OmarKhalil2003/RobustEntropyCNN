# RobustEntropyCNN
## Overview

This project extends the paper **“Entropy-Based Non-Invasive Reliability Monitoring of Convolutional Neural Networks”** by transforming entropy from a *purely observational signal* into an **actionable reliability mechanism**.

While the original paper demonstrates that entropy correlates with prediction reliability under adversarial perturbations, it remains architecture-limited and does not operationalize entropy for decision-making.
This work addresses those gaps through **architecture-aware validation, hybrid CNN design, selective classification, and stronger adversarial evaluation**.

---

## Paper Motivation

### Problem

* CNNs are highly vulnerable to adversarial attacks (FGSM, PGD), often producing **confident but incorrect predictions**.
* Existing robustness defenses typically require **retraining** or significantly degrade clean accuracy.
* The referenced paper proposes entropy-based monitoring but:

  * Evaluates a **single CNN architecture**
  * Uses entropy only as a **diagnostic signal**
  * Focuses mainly on **FGSM attacks**

### Paper Hypothesis

> Internal entropy patterns inside CNNs provide a non-invasive signal that correlates with prediction reliability under adversarial perturbations.

---

## Project Contributions

This project **does not challenge the correctness of the paper**, but **extends it empirically and practically**:

### Key Improvements

* **Architecture-aware validation**
  Tested entropy behavior across:

  * Simple CNN (baseline)
  * MobileNet
  * Hybrid CNN (depthwise + grouped + attention)

* **Entropy → Decision Control**
  Converted entropy from:

  > “This prediction might be wrong”
  > to:
  > “This prediction should not be trusted or used”

* **Selective Classification**

  * Entropy-based rejection thresholds
  * Explicit coverage vs accuracy trade-off
  * Reliability-aware inference

* **Stronger Adversarial Evaluation**

  * FGSM (single-step)
  * PGD (iterative, stronger attack)

* **Hybrid Architecture Design**

  * Transfer learning (MobileNetV2 backbone)
  * Grouped & depthwise convolutions
  * Channel-wise attention (SE blocks)
  * Optional 3D path for volumetric data

---

## Project Structure

```
RobustEntropyCNN/
│
├── models/
│   ├── base_models/
│   │   ├── simple_cnn.py
│   │   ├── mobilenet_baseline.py
│   │   └── hybrid_cnn.py
│   │
│   └── layers/
│       ├── depthwise_blocks.py
│       ├── grouped_conv.py
│       └── attention_se.py
│
├── attacks/
│   ├── fgsm.py
│   └── pgd.py
│
├── evaluation/
│   ├── entropy.py
│   ├── selective_classification.py
│   └── metrics.py
│
├── experiments/
│   ├── train.py
│   ├── run_attack_pipeline.py
│   └── run_visualization_pipeline.py
│
├── saved_models/
│
└── README.md
```

---

## Experimental Phases

### Phase 1 — Baseline Validation

* Simple CNN and MobileNet
* Clean accuracy is high
* Accuracy collapses under FGSM/PGD
* Entropy correlates with failure but **does not prevent it**

**Key insight:**
High clean accuracy ≠ robustness

---

### Phase 2 — Hybrid Architecture

* MobileNetV2 pretrained backbone
* Domain adaptation layer (1×1 conv)
* Hybrid feature extraction:

  * Grouped or depthwise convolutions
* SE attention blocks
* Multi-domain support (2D / 3D)

**Results:**

* Significant improvement in clean accuracy
* More stable entropy distributions
* Better selective classification behavior
* Still vulnerable to strong adversarial attacks

---

### Phase 3 — Reliability Evaluation

* Side-by-side comparison: Phase 1 vs Phase 2
* Metrics:

  * Clean accuracy
  * FGSM accuracy
  * PGD accuracy
  * Selective accuracy vs coverage
  * Latency

**Conclusion:**
Entropy **detects failures reliably** but **does not provide robustness by itself**.

---

## Key Results Summary

| Metric             | Phase 1 (Baseline) | Phase 2 (Hybrid) |
| ------------------ | ------------------ | ---------------- |
| Clean Accuracy     | ~73.7%             | ~85.5%           |
| FGSM Accuracy      | ~1.5%              | ~10%             |
| PGD Accuracy       | ~0%                | ~10%             |
| Selective Accuracy | 73.6% @97%         | 85.4% @98.7%     |

---

## Main Conclusions

* Entropy is a **reliable uncertainty signal**
* Stronger architectures produce **cleaner entropy separation**
* Entropy alone **does not guarantee robustness**
* Selective classification converts uncertainty into **practical reliability control**
* Robustness–accuracy trade-off remains unresolved without training-time defenses

---

## Limitations

* No adversarial training applied
* Fixed entropy thresholds
* Limited real-world datasets (e.g., autonomous driving not yet tested)

---

## Future Work

* Integrate entropy with adversarial training
* Learn adaptive, architecture-aware entropy thresholds
* Fuse multi-layer entropy signals
* Evaluate in safety-critical real-world systems
* Combine entropy with complementary uncertainty metrics

---

## Final Judgment

> This project extends entropy-based monitoring from a **conceptual diagnostic tool** into a **deployable reliability framework**, addressing architectural diversity, decision-making, and adversarial severity — without contradicting the original paper’s claims.

---
##  Author

**Omar Khalil**
MSc Researcher @ AASTMT
Connect on [LinkedIn](https://www.linkedin.com/in/omar-khalil-10om01) or check more projects!

---

## License

This project is for **Research purposes**.
All dataset and model weights belong to their respective owners.

---
