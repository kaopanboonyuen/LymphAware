# 🧬 LymphAware: Domain-Aware Bias Disruption for Reliable Lymphoma Cancer AI Diagnosis

<div align="center">

<b>Teerapong Panboonyuen</b><br/>
College of Computing, Khon Kaen University<br/>
🎓 Supported by the <b>Talent Scholarship for Exceptional Ability</b><br/><br/>

<b>Peer-Reviewed & Accepted at IEEE Access (February 2026)</b> 🎉<br/>
DOI: 10.1109/ACCESS.2026.3667575

</div>

---

<p align="center">
  <img src="images/main_01.png" width="95%" alt="LymphAware Architecture"/>
</p>

## 🚀 Overview

**LymphAware** is a domain-aware bias disruption framework designed to improve the **reliability, robustness, and clinical relevance** of AI systems for lymphoma histopathology diagnosis.

Modern medical AI models often achieve high accuracy by exploiting **non-biological shortcuts** — such as stain color, scanner signatures, or slide artifacts — instead of true pathological morphology. While effective in-domain, these shortcuts lead to **fragile performance under cross-center variability**, which is unacceptable for clinical deployment.

LymphAware explicitly addresses this challenge by **separating morphology-relevant signals from shortcut-driven acquisition factors**, enabling models to “think more like pathologists.” 🧠🔬

---

## ✨ Key Innovations

🔹 **Tri-Path Morphology Purification Architecture**

* Morphology-centric feature encoder
* Shortcut identification & suppression branch
* Cross-domain stability alignment stream

🔹 **Artifact-Shift Counterfactual Training**

* Simulated staining and scanner perturbations
* Exposure of latent shortcut dependencies
* Acquisition-invariant representation learning

🔹 **Domain-Aware Robustness Without Explicit Labels**

* Works under realistic multi-source settings
* No assumption of verified institutional separation

---

## 📊 Qualitative Results — Shortcut Suppression

<p align="center">
  <img src="images/main_02.png" width="95%" alt="Qualitative Results"/>
</p>

Models trained **without** LymphAware rely heavily on stain tone, background artifacts, and acquisition noise.
With LymphAware, attention shifts toward **diagnostically meaningful lymphoid morphology**.

---

## 📈 Cross-Center Performance

<p align="center">
  <img src="images/main_03.png" width="95%" alt="Performance Tables"/>
</p>

Across five independent medical centers:

✅ Higher AUC
✅ Lower false positive rates
✅ Reduced variance across backbones
✅ Stronger causal consistency metrics

---

## 🏆 Acceptance Evidence

<p align="center">
  <img src="images/main_04.png" width="95%" alt="IEEE Acceptance"/>
</p>

This work has been **peer-reviewed and accepted** for publication in *IEEE Access*, highlighting its contribution to reliable medical AI research.

---

## 📖 Official Publication & Citation

<div align="center">

🔗 **Read the official IEEE publication:**
[https://ieeexplore.ieee.org/document/11408775](https://ieeexplore.ieee.org/document/11408775)

📌 **DOI:** 10.1109/ACCESS.2026.3667575

Published in *IEEE Access*, Early Access, February 2026.

</div>

---

## 🧠 Why LymphAware Matters

Medical AI systems must be:

* ✔ Robust across scanners and hospitals
* ✔ Grounded in biological morphology
* ✔ Clinically interpretable
* ✔ Stable under domain shift

LymphAware moves the field closer to **trustworthy computational pathology** by addressing shortcut bias at the **representation level**, rather than relying solely on dataset curation or domain labels.

---

## 🚀 Training LymphAware

We provide a **clean, reproducible PyTorch pipeline** located in the `src/` directory for training LymphAware across multi-center lymphoma datasets.

The framework is **backbone-agnostic** and supports:

* 🧠 ResNet (18 / 50 / 152)
* 🌿 DenseNet (121)
* 🔭 Vision Transformers (ViT-L/16)
* 🏥 Multi-center domain training (Centers A–E)
* 🎨 Artifact-shift augmentation for shortcut exposure
* 📈 AUC and FPR evaluation

---

### 📂 Project Structure

```
LymphAware/
│
├── src/
│   ├── train_lymphaware.py
│   ├── models/
│   ├── datasets/
│   ├── losses/
│   └── utils/
│
├── data/
│   ├── CenterA/
│   ├── CenterB/
│   ├── CenterC/
│   ├── CenterD/
│   └── CenterE/
│
└── outputs/
```

Each center directory should contain class folders:

```
CenterA/
    CLL/
    FL/
    MCL/
```

---

### ⚙️ Installation

```bash
git clone https://github.com/kaopanboonyuen/LymphAware.git
cd LymphAware

conda create -n lymphaware python=3.10
conda activate lymphaware

pip install -r requirements.txt
```

---

### ▶️ Training Example

Train on a specific center (e.g., Center A):

```bash
python src/train_lymphaware.py \
    --train_dir data/CenterA/train \
    --val_dir data/CenterA/test \
    --backbone resnet50 \
    --epochs 100 \
    --batch_size 16 \
    --lr 3e-4
```

---

### 🔬 Training with Vision Transformer (Best Performance)

```bash
python src/train_lymphaware.py \
    --train_dir data/CenterA/train \
    --val_dir data/CenterA/test \
    --backbone vit_large_patch16_224 \
    --epochs 100
```

---

### 💾 Outputs

Training artifacts will be saved to:

```
outputs/
    best_model.pth
```

The script automatically:

✅ Tracks validation AUC
✅ Computes False Positive Rate (FPR)
✅ Saves the best checkpoint
✅ Supports GPU acceleration

---

### 🧪 Multi-Center Reproduction (Centers A–E)

To reproduce the paper results:

1. Train a model per center
2. Evaluate cross-domain performance
3. Average metrics across runs

Example loop:

```bash
for CENTER in CenterA CenterB CenterC CenterD CenterE
do
  python src/train_lymphaware.py \
      --train_dir data/${CENTER}/train \
      --val_dir data/${CENTER}/test \
      --backbone vit_large_patch16_224
done
```

---

### ⭐ Research Tips (From the Paper)

For best performance reported in IEEE Access:

* Backbone: **ViT-L/16**
* Epochs: **100**
* Optimizer: **AdamW**
* Learning rate: **3e-4**
* Image size: **224 × 224**
* Loss weight (orthogonality): **0.1**

---

### 🧠 Why This Training Matters

Unlike standard pipelines, LymphAware training:

* Disrupts shortcut bias during representation learning
* Encourages morphology-grounded predictions
* Improves robustness across scanners and institutions
* Produces clinically meaningful attribution behavior

> **The model learns cancer morphology — not acquisition artifacts.**

---

If you find this work useful, please ⭐ star the repository.

---

## 🙏 Acknowledgement

This research is supported by:

🎓 Talent Scholarship for Exceptional Ability
🏫 College of Computing, Khon Kaen University

---

## 🌟 Final Note

> **LymphAware learns the cancer — not the confounders.**

By enforcing morphology-grounded representations and suppressing shortcut bias, we aim to build AI systems that clinicians can truly trust.

---

⭐ If you find this project useful, please consider starring the repository!

---

### 📚 BibTeX Citation

```bibtex
@article{panboonyuen2026lymphaware,
  author    = {Teerapong Panboonyuen},
  title     = {LymphAware: Domain-Aware Bias Disruption for Reliable Lymphoma Cancer AI Diagnosis},
  journal   = {IEEE Access},
  year      = {2026},
  pages     = {1--1},
  doi       = {10.1109/ACCESS.2026.3667575},
  publisher = {IEEE}
}
```

If you use this work in your research, please cite the official IEEE version via the DOI above.

---