# 🩺 SPDS_XAI – Smart Pneumonia Detection System with Explainable AI

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📑 Table of Contents
1. [Overview](#overview)
2. [Abstract](#abstract)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [System Architecture](#system-architecture)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Example Output](#example-output)
9. [Dataset](#dataset)
10. [Future Enhancements](#future-enhancements)
11. [License](#license)
12. [Contact](#contact)

---

## 📌 Overview
**SPDS_XAI** is a deep learning-based medical imaging tool designed to detect **Pneumonia** from chest X-ray images.  
It integrates **Explainable AI (XAI)** techniques, specifically **Saliency Maps**, to highlight important regions in the X-ray that influenced the model's decision.  
This enhances trust and transparency in AI-driven medical diagnostics.

---

## 📜 Abstract
Pneumonia is a life-threatening respiratory infection that requires timely detection and treatment.  
Manual diagnosis from chest X-rays can be time-consuming and subject to human error.  
The **Smart Pneumonia Detection System with Explainable AI (SPDS_XAI)** uses convolutional neural networks to automate detection while maintaining interpretability through **Saliency Maps**.

The system:
- Accepts chest X-ray images as input.
- Preprocesses and classifies them into *Pneumonia* or *Normal*.
- Generates **visual explanations** to show which parts of the image contributed most to the diagnosis.
- Outputs results in both on-screen display and a downloadable PDF report.

This approach supports medical practitioners by providing **fast, transparent, and accurate** AI assistance.

---

## ✨ Features
- 🩻 **Pneumonia Detection** from chest X-rays.
- 🔍 **Explainability** via Saliency Maps.
- 📄 **PDF Report Generation**.
- 💻 **Offline Processing** — no internet needed.
- ⚡ **High Accuracy** using deep learning.

---

## 🛠 Tech Stack
- **Programming Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** OpenCV, Pillow
- **Explainability:** Matplotlib (custom saliency map)
- **PDF Reports:** ReportLab / FPDF
- **Environment:** Local execution (no cloud dependency)

---

## 🏗 System Architecture
                          [User Input: X-ray Image]
                                      │
                                      ▼
                            [Image Preprocessing]
                                      │
                                      ▼
                          [Deep Learning Model: CNN]
                                      │
                    ├── Prediction (Pneumonia / Normal)
                    └── Saliency Map Generation
                                      │
                                      ▼
                            [PDF Report Creation]
                                      │
                                      ▼
                    [Output: Prediction + Heatmap + Report]

---

## ⚙ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/akshithaa016/SPDS_XAI.git
cd SPDS_XAI

2️⃣ Create Virtual Environment

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows


3️⃣ Install Dependencies

pip install -r requirements.txt


Place your chest X-ray image in the images/ folder.

python src/predict.py --image images/sample.jpg
The output will include:

Predicted class: Pneumonia / Normal

Probability score

Saliency Map

PDF Report in reports/ folder

📊 Example Output
Original X-ray	Saliency Map Overlay

📂 Dataset
This project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle:
🔗 https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

🚀 Future Enhancements
Add Grad-CAM for improved heatmaps.

Extend model to detect multiple lung diseases.

Deploy as a Streamlit/Flask web application.

Implement medical data compliance (HIPAA/GDPR).

📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

📬 Contact
👤 Akshitha
📧 Email: [akulaakshitha016@gmail.com]
🔗 GitHub: https://github.com/<akshithaa016>
