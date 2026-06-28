# AI Face Shape Analysis & Intelligent Hairstyle Recommendation System

> An AI-powered face shape analysis and hairstyle recommendation system based on **ResNet34**, **MediaPipe FaceMesh**, and an explainable **Recommendation Knowledge Base**.

---

## 📌 Project Overview

This project aims to build a complete AI face shape analysis and intelligent hairstyle recommendation system instead of simply comparing deep learning models.

The system combines **ResNet34** for face shape classification, **MediaPipe FaceMesh** for facial landmark analysis, a **Geometry Refinement** module for low-confidence predictions, and a **Knowledge Base** with explainable recommendation rules to provide reliable hairstyle suggestions.

---

## ✨ Features

* Face Shape Classification (ResNet34)
* Face Landmark Analysis (MediaPipe FaceMesh)
* Low-confidence Geometry Refinement
* Analysis Confidence Report
* Explainable Hairstyle Recommendation
* Recommendation Knowledge Base
* Hairstyle Reference Gallery
* Dark / Light Theme

---

## 🏗️ System Pipeline

Upload Image

→ Face Detection

→ Landmark Detection

→ Face Shape Classification (ResNet34)

→ Low-confidence Geometry Refinement

→ Recommendation Knowledge Base

→ Hairstyle Recommendation

→ Analysis Result

---

## 📂 Project Structure

```text
project/
│
├── app.py
├── core/
├── ui/
├── assets/
├── style.css
├── hairstyle_rules.json
├── requirements.txt
└── README.md
```

---

## 🧠 Technologies

* Python
* PyTorch
* Gradio
* MediaPipe
* OpenCV
* Pillow
* NumPy

---

## 📂 Dataset

The face shape classification model in this project was primarily trained using the public Face Shape Dataset from Kaggle.

Dataset Information
Dataset Name: Face Shape Dataset
Source: Kaggle
URL: https://www.kaggle.com/datasets/niten19/face-shape-dataset
Face Shape Classes:
Oval
Round
Square
Heart
Oblong
Dataset Notes

The current dataset is mainly composed of publicly available face images collected from the web, with the majority representing Western facial features. Therefore, the trained model may have limited generalization ability for Asian faces.

To ensure transparency and reliability, this project does not claim that the model is equally applicable to all ethnic groups. Expanding and validating the dataset with more Asian face samples is one of the primary future improvement goals.

---

## ⚠️ Limitations

* The training dataset is mainly composed of publicly available Western face images, which may limit the model's generalization ability to Asian faces.
* The system is designed primarily for frontal face images. Large head rotations, occlusions, or poor lighting conditions may reduce prediction reliability.
* Facial geometry analysis is intended as an explainable auxiliary feature and a low-confidence refinement mechanism rather than the primary classification method.
* The upper-face width is estimated from MediaPipe FaceMesh landmarks and does not represent the actual hairline or full forehead.
* Hairstyle recommendations are currently generated using a rule-based knowledge base and have not yet incorporated personalized user preferences.

---

## 🚀 Future Work

* Build an Asian face shape evaluation dataset.
* Improve recommendation quality using user preferences.
* Add photo quality assessment.
* Deploy the system online.
