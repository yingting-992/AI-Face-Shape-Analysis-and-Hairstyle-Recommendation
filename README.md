# AI 臉型分析與智慧髮型推薦系統

> 整合深度學習、人臉分析與可解釋推薦的 AI Computer Vision 應用系統

---

## 專案介紹

本專案旨在建立一套完整的 **AI 臉型分析與智慧髮型推薦系統**，整合深度學習、人臉分析與髮型推薦知識庫，提供具有可解釋性的髮型推薦結果。

不同於單純進行臉型分類，本系統更重視完整的 AI 應用流程，包含圖片上傳、人臉偵測、臉部校正、臉型辨識、信心值分析、髮型推薦、推薦理由與結果展示，使系統更接近實際應用情境。

本專案定位為一套 **AI Computer Vision 應用系統**，主要作為 GitHub 作品集、研究所推甄及求職展示使用，而非單純的模型比較研究。

---

## 專案特色

- 人臉偵測與前處理
- 臉部 Landmark 偵測
- 臉型辨識（ResNet34）
- 信心值分析
- 幾何特徵補判機制
- 髮型推薦
- 推薦理由說明
- 髮型圖片展示
- Web 操作介面（Gradio）

---

## 系統流程

```text
圖片上傳
    │
    ▼
人臉偵測
    │
    ▼
Landmark 偵測
    │
    ▼
人臉校正
    │
    ▼
ResNet34 臉型辨識
    │
    ▼
信心值分析
    │
    ▼
低信心幾何補判
    │
    ▼
髮型推薦
    │
    ▼
推薦理由
    │
    ▼
結果展示
```

---

## 臉型辨識模型

目前使用模型：

- ResNet34（Transfer Learning）

訓練流程包含：

- Transfer Learning
- Data Augmentation
- Grid Search
- Label Smoothing
- Class Weight
- Learning Rate Scheduler
- Early Stopping
- Confusion Matrix
- ROC Curve
- Classification Report
- Macro F1 Score

---

## 髮型推薦模組

目前推薦依據包含：

- 臉型分類結果
- 推薦知識庫
- 推薦理由
- 推薦圖片

後續將持續加入：

- 臉部比例
- 瀏海類型
- 髮長
- 髮量
- 捲度
- 使用者偏好

---

## 資料集

目前模型使用公開 Face Shape Dataset 訓練。

分類包含：

- Heart
- Oblong
- Oval
- Round
- Square

資料前處理包含：

- 人臉裁切
- 影像縮放
- Data Augmentation
- Train / Validation / Test Split

---

## 系統限制

目前模型主要使用公開 Face Shape Dataset 訓練，資料來源以歐美臉孔為主，亞洲臉孔比例不足，因此模型在亞洲使用者上的泛化能力仍需進一步驗證。

本系統分析結果主要提供髮型推薦參考，不應視為絕對臉型判定。

---

## 未來規劃

- 建立亞洲臉型驗證資料集
- 擴充髮型推薦知識庫
- 加入臉部比例分析
- 加入瀏海、髮長、髮量等推薦條件
- 提升推薦可解釋性
- 完善系統文件
- 增加 Demo 展示

---

## 安裝方式

```bash
pip install -r requirements.txt
```

---

## 執行方式

```bash
python app.py
```