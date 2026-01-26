# 🚀 DeepSight: OSEMN Transfer Learning & Forensic Diagnostics

## 📌 Project Overview
**DeepSight** is a high-performance computer vision pipeline designed to master the CIFAR-10 classification challenge through the **OSEMN framework**. Beyond standard predictive modeling, this project functions as a **Forensic Diagnostic Tool**, leveraging **Explainable AI (XAI)** to validate model decision-making processes and mitigate systemic issues such as background bias and semantic ambiguity.

---

## 🛠 Technical Workflow (The OSEMN Approach)

### 1. Obtain: Data Acquisition & Stratification

* **Modular Ingestion:** Seamless CIFAR-10 integration via a decoupled `obtain.py` module for improved maintainability.
* **Stratified Sampling:** Implementation of a **10% stratified subset** to facilitate rapid prototyping while preserving original class balance.
* **MVA Baseline:** Establishment of a **Minimum Viable Accuracy (MVA)** threshold of 65% for the initial frozen-base development phase.

### 2. Scrub: Data Engineering & Normalization
* **Dimensional Consistency:** Automated resizing pipelines to $(3, 224, 224)$ to satisfy **ResNet-18** architectural input requirements.
* **Interpolation Strategy:** Utilization of **Bicubic Interpolation** to preserve high-frequency spatial details, outperforming standard bilinear methods in feature retention.
* **Feature Scaling:** Application of **ImageNet-specific normalization** ($\mu, \sigma$) to align input data with the pre-trained manifold.

### 3. Explore: Forensic EDA
* **Spatial Analysis:** Evaluation of **Annotation Density** to identify and account for center-bias inherent in the CIFAR-10 dataset.
* **Semantic Ambiguity:** Mapping the **"Car-Truck Boundary"** to anticipate and mitigate class-overlap during the subsequent modeling phase.
* **Hypothesis Testing:** Investigation of color-channel distributions to identify potential **"Sky-Bias"** within airplane vs. bird classifications.

### 4. Model: Two-Phase Transfer Learning

* **Phase 1 (Feature Extraction):** Deployment of a frozen **ResNet-18 backbone** featuring a custom-reconstructed **Fully Connected (FC) head**.
* **Phase 2 (Fine-Tuning):** Strategic unfreezing of **Stage 3 & 4** of the backbone to allow for domain-specific weight adaptation.
* **Optimization Strategy:** Implementation of **Differential Learning Rates** to balance stability and adaptation:
    * **Backbone Layers:** $\eta = 10^{-5}$
    * **Classification Head:** $\eta = 10^{-4}$

### 5. iNterpret: Explainable AI (XAI)

* **Forensic Diagnostics:** Generation of dual **Confusion Matrices** to visualize the performance delta between Phase 1 (Frozen) and the Final Fine-Tuning stage.
* **Explainability:** Leveraging **Grad-CAM** (Gradient-weighted Class Activation Mapping) to produce heatmaps, verifying that the model prioritizes **global shapes** over local texture leaks or background noise.

---

## 📊 Performance Benchmarks

| Metric | Phase 1 (Frozen) | Final (Fine-Tuning) |
| :--- | :--- | :--- |
| **Accuracy** | 66.5% | **92.0% 🏆** |
| **Macro F1-Score** | 0.67 | 0.92 |
| **Precision (Truck)** | 0.72 | 0.96 |
| **Inference Mode** | Feature Extraction | Weight Adaptation |

> [!NOTE]
> The **26.5% accuracy increase** observed between Phase 1 and Phase 2 confirms the successful adaptation of deep convolutional filters to CIFAR-10 specific visual textures.

---

## 🔥 Visual Evidence (XAI)

* **Grad-CAM:** Diagnostic plots located in the `notebooks/` directory confirm a **90%+ object-focus rate**, indicating high model reliability.
* **Confusion Matrix:** Final analysis demonstrates a significant reduction in **"Semantic Swapping"** between rigid vehicle classes (e.g., Cars and Trucks).

---
