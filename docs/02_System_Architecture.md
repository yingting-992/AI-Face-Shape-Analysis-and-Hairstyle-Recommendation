# System Architecture

## Overall System Architecture

The system is designed as a complete AI application rather than a standalone face shape classification model.

The workflow consists of several independent but connected modules, allowing each component to be improved or replaced independently.

```text
                  User
                    │
                    ▼
            Image Upload
                    │
                    ▼
            Face Detection
                    │
                    ▼
      Face Alignment & Preprocessing
                    │
                    ▼
      Face Shape Classification Model
                    │
                    ▼
          Confidence Analysis
                    │
                    ▼
      Hairstyle Recommendation Engine
                    │
                    ▼
      Recommendation Knowledge Base
                    │
                    ▼
     Recommendation Reason Generator
                    │
                    ▼
         Result Visualization
```

## System Modules

### Image Upload

Users upload a facial image through the web interface.

---

### Face Detection

Detect the facial region from the original image and remove unnecessary background information.

---

### Face Alignment

Align the face to reduce variations caused by head rotation or camera angle.

---

### Face Shape Classification

The aligned face image is sent to the deep learning model for classification.

Current implementation:

* ResNet34
* Transfer Learning

---

### Confidence Analysis

The system outputs the confidence score for every face shape category.

Low-confidence predictions can be highlighted for further analysis or future improvement.

---

### Hairstyle Recommendation Engine

The recommendation engine combines face shape information with the hairstyle knowledge base.

Instead of using a fixed mapping table, future versions will consider multiple facial and hairstyle attributes.

---

### Recommendation Knowledge Base

The knowledge base stores hairstyle characteristics, including:

* Suitable face shapes
* Hair length
* Bang styles
* Hair layers
* Hair volume
* Curl types
* Recommendation reasons

---

### Recommendation Reason Generator

The system explains why each hairstyle is recommended, improving transparency and user trust.

---

### Result Visualization

Finally, the system presents:

* Predicted face shape
* Confidence score
* Recommended hairstyles
* Recommendation reasons
* Related hairstyle examples
