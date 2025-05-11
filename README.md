# ✈️ EVA_LAND – Explainable Vision-based Assistance Landing

**EVA_LAND** is a real-time AI co-pilot system that supports emergency aircraft landings through explainable computer vision.

This Phase 1 implementation uses deep learning and explainability techniques to detect viable landing zones and interpret AI decisions in a human-understandable way.

---

## 🧠 Core Technologies

- **CNN (Convolutional Neural Networks)** – for binary runway detection (runway vs non-runway)
- **Grad-CAM** – to visualize which regions influenced the AI's decision
- **Perturbation Analysis + Faithfulness Check** – to verify explanation reliability
- **Gradio** – for interactive, real-time testing and demonstration

---

## 📦 Dataset

We trained EVA_LAND on the **Landing Approach Runway Detection (LARD)** dataset:

- Aerial front-view images during aircraft approaches
- Contains both runway and non-runway classes

📁 Dataset Source:
- [Data Archive](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/MZSH2Y)  
- [Official GitHub](https://github.com/deel-ai/LARD)  
- [Publication (ArXiv)](https://arxiv.org/abs/2304.09938)


