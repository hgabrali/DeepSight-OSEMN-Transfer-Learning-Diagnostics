#  DeepSight: OSEMN Transfer Learning & Forensic Diagnostics




## 📌 Project Overview 
Transitioning from the grayscale simplicity of datasets like Fashion MNIST, this project tackles the **CIFAR-10** dataset—a benchmark in computer vision consisting of 60,000 $32 \times 32$ color images across 10 mutually exclusive classes.

The primary objective is to implement a **Transfer Learning** approach using the **ResNet18** architecture, demonstrating the efficiency of using pre-trained weights (ImageNet) to solve complex image classification tasks even with relatively low-resolution inputs. Unlike standard implementations, this project enforces a **Minimum Viable Accuracy (MVA)** of 65% and employs a rigorous **"Forensic Error Analysis"** pipeline to detect semantic ambiguity and contextual bias.

* I feel like 'DeepSight' is such a spot-on metaphor. It’s basically where Computer Vision and Deep Learning meet. Using OSEMN as my roadmap, I decided to go with Transfer Learning—which meant taking models like ResNet or VGG that were already trained on massive datasets and just fine-tuning them for my needs.*  
---

## 📊 Dataset Insights & Problem Framing
CIFAR-10 presents a unique challenge due to the low signal-to-noise ratio inherent in its resolution.



* **Dimensions:** $32 \times 32$ pixels, 3 Channels (RGB).
* **Scale:** 50,000 Training samples, 10,000 Testing samples.
* **Classes:** ✈️ Airplane, 🚗 Automobile, 🐦 Bird, 🐱 Cat, 🦌 Deer, 🐶 Dog, 🐸 Frog, 🐴 Horse, 🚢 Ship, 🚛 Truck.
* **Semantic Nuance:** The dataset contains inherent ambiguity; for instance, the "Automobile" class includes sedans/SUVs but excludes pickup trucks (which fall under "Truck"). This requires the model to learn subtle geometric boundary features rather than just general shapes.

---

## 🛠 Methodology: The OSEMN Framework
This project follows the **OSEMN** (Obtain, Scrub, Explore, Model, iNterpret) pipeline to ensure reproducibility and engineering rigor.

<img width="1238" height="678" alt="image" src="https://github.com/user-attachments/assets/5e0225dc-dc0b-41b6-a97c-12c5ef5bac61" />


### 1. Phase 1: Data Acquisition (Obtain)
* **Ingestion:** Data is loaded via `tensorflow.keras.datasets` to ensure source integrity.
* **Orchestration:** The pipeline is orchestrated using **PyTorch** for Deep Learning and **Albumentations** for robust data augmentation.
* **Metric Definition:** A custom success metric, **Macro-Averaged F1-Score**, is defined to mitigate bias in potentially unbalanced subsets.

### 2. Phase 2: Data Engineering (Scrub)
* **Dimensional Integrity:** A pre-batch verification function asserts input tensor shapes $(3, 32, 32)$ to prevent runtime shape mismatches.
* **Interpolation Strategy:** A comparative analysis between Bilinear and Bicubic resizing was conducted. **Bicubic Interpolation** was selected as the standard to preserve high-frequency details (textures) required for the ResNet backbone.
* **Leakage Prevention:** A `StratifiedShuffleSplit` is applied to strictly segregate the test set, ensuring no samples overlap and class distributions remain uniform.

### 3. Phase 3: Exploratory Data Analysis (Explore)
* **Class Balance:** Visual analysis confirms a flat frequency distribution, ensuring no single class dominates the loss gradient.
* **Hypothesis Generation:**
    * **Rigid vs. Organic:** We hypothesized that rigid objects (Cars, Ships) would yield higher accuracy due to distinct geometric edges, whereas organic objects (Cats, Deer) would suffer from texture confusion.
    * **Semantic Ambiguity:** Visual inspection revealed potential overlap between "Automobile" and "Truck" classes at low resolutions due to similar chassis structures.

### 4. Phase 4: Model Architecture (Model)

* **Backbone:** **ResNet18** (Pre-trained on ImageNet) is utilized to leverage robust hierarchical feature extractors (edges, textures).
* **Head Reconstruction:** The final Fully Connected (FC) layer is replaced to project the 1,000 ImageNet classes down to the 10 CIFAR classes.
* **Training Dynamics:**
    * **Frozen Phase:** The backbone is frozen to prevent **"Catastrophic Forgetting"** of pre-trained weights during the initial epochs.
    * **Fine-Tuning:** Deep layers (Stage 3 & 4) are unfrozen to allow **Domain Adaptation** for the $32 \times 32$ pixel space.
    * **Hyperparameters:** Utilization of **Differential Learning Rates** (lower for base, higher for head) and a `OneCycleLR` scheduler for super-convergence.

### 5. Phase 5: Diagnostic Evaluation (iNterpret)
We move beyond "Black Box" metrics to determine if the model learned features or simply memorized noise.

#### A. Quantitative Analysis

* **Confusion Matrix:** Used to identify specific cluster errors. Analysis revealed "Cat vs. Dog" (texture similarity) and "Plane vs. Bird" (background bias) as primary confusion vectors.
* **Class-Wise Accuracy:** A decomposition of recall per class. Rigid objects generally outperformed organic ones, validating our initial hypothesis regarding "Rigid vs. Deformable" objects.

#### B. Visual Interpretability (XAI)

* **Grad-CAM (Gradient-weighted Class Activation Mapping):**
    * **Objective:** To visualize the "Attention Mechanism" of the CNN.
    * **Forensic Result:** Heatmaps verify that the model focuses on the physical object (e.g., the fuselage of a plane) rather than the background (e.g., blue sky), confirming the absence of **Contextual Overfitting**.

---

## 🚀 Key Takeaways & Conclusion
* **Generalization Verified:** The correlation between input objects and Grad-CAM activation peaks proves the model operates as a valid feature extractor.
* **Semantic Bottlenecks:** The primary sources of error are **Semantic Overlap** (e.g., Cat/Dog fur textures) rather than data quality issues.
* **Future Interventions:**
    * **Hard Example Mining:** Retrain specifically on confused pairs (e.g., create a batch containing only Cats and Dogs).
    * **CutMix Augmentation:** Apply advanced augmentation to force the model to focus on structural boundaries rather than just texture.

---

## 👨‍🏫  Post-Analysis (Next Steps)
* **Address the "Cat-Dog" Problem:** The Confusion Matrix highlights a texture bias. Implement **CutMix** or **MixUp** augmentation to force the model to learn shape boundaries.
* **Re-Verify ResNet Depth:** Explicitly verify if **ResNet18** performs better for $32 \times 32$ images, as ResNet50 may be over-parameterized for this specific resolution.
* **Weighted Loss:** If specific classes are underperforming, switch to a **Weighted CrossEntropyLoss** to assign higher penalties to weak classes (Cats, Birds).

---

