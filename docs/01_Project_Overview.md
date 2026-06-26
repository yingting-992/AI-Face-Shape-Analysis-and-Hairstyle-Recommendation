# AI Face Shape Analysis & Intelligent Hairstyle Recommendation System

An AI-powered computer vision system for face shape analysis and explainable hairstyle recommendation.

## Overview

本專案旨在建立一套完整的 **AI 臉型分析與智慧髮型推薦系統**，整合深度學習、人臉分析與髮型推薦知識庫，提供具有可解釋性的髮型推薦結果。

不同於單純進行臉型分類，本系統更重視完整的 AI 應用流程，包含圖片上傳、人臉偵測、臉部校正、臉型辨識、信心值分析、髮型推薦、推薦理由與結果展示，使系統更接近實際應用情境。

本專案定位為一套 **AI Computer Vision Application**，主要作為 GitHub 作品集、研究所推甄及求職展示使用，而非單純的模型比較研究。

## Features

* Face detection and preprocessing
* Face shape classification
* Confidence score analysis
* Explainable hairstyle recommendation
* Hairstyle recommendation knowledge base
* Recommendation reason generation
* Result visualization
* Web demo interface

## System Pipeline

```text
Image Upload
     |
     v
Face Detection
     |
     v
Face Alignment / Preprocessing
     |
     v
Face Shape Classification
     |
     v
Confidence Analysis
     |
     v
Hairstyle Recommendation Engine
     |
     v
Recommendation Reason Generation
     |
     v
Result Visualization
```

## Face Shape Classification

The face shape classification model is based on transfer learning and deep convolutional neural networks.

Current model:

* ResNet34

Training and evaluation process includes:

* Transfer Learning
* Data Augmentation
* Grid Search
* Label Smoothing
* Class Weight
* Learning Rate Scheduler
* Early Stopping
* Confusion Matrix
* ROC Curve
* Classification Report
* Macro F1 Score

## Hairstyle Recommendation

The recommendation module is designed to avoid a simple fixed mapping such as:

```text
Round face -> Hairstyle A
Oval face  -> Hairstyle B
```

Instead, the system aims to provide explainable recommendations based on multiple factors, including:

* Face shape
* Hair length
* Bang style
* Hair volume
* Layer style
* Curl type
* Facial proportion
* User preference

Each recommendation will include not only hairstyle suggestions, but also the reason why the hairstyle may be suitable.

## Dataset

The current model is trained using a public Face Shape Dataset.

Face shape categories include:

* Heart
* Oblong
* Oval
* Round
* Square

The dataset processing workflow includes:

* Image cleaning
* Face cropping
* Image resizing
* Data augmentation
* Train / validation / test split

## Current Limitations

The main limitation of this project is the dataset source.

The current public Face Shape Dataset mainly consists of Western facial images. Therefore, the model may learn facial features that are more representative of Western users, and its generalization ability for Asian users has not yet been fully verified.

This project does not claim to be universally applicable to all ethnic groups at the current stage.

Future improvements will focus on:

* Building an Asian face shape testing set
* Collecting additional Asian validation samples
* Improving dataset diversity
* Evaluating cross-ethnic generalization ability
* Improving recommendation reliability

## Project Structure

```text
AI-Face-Shape-Analysis-and-Hairstyle-Recommendation/
│
├── README.md
├── requirements.txt
│
├── assets/
├── datasets/
├── docs/
│   ├── 01_Project_Overview.md
│   ├── 02_System_Architecture.md
│   ├── 03_Dataset.md
│   └── 04_Model.md
│
├── inference/
├── models/
├── recommendation/
├── results/
├── training/
└── web/
```

## Future Work

Planned improvements include:

* Building a more complete recommendation knowledge base
* Adding facial proportion analysis
* Improving low-confidence prediction handling
* Expanding Asian face shape validation data
* Developing an interactive web demo
* Adding hairstyle visualization examples
* Improving documentation and deployment process

## Project Status

This project is currently under development.

The current focus is not only improving classification accuracy, but also building a complete, reliable, and explainable AI application system.
