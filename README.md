# Facial Emotion Recognition Using FER2013

A machine learning and deep learning project for recognizing human facial emotions from facial images using the **FER2013 dataset**. The project compares traditional statistical pattern recognition techniques using **HOG, PCA, and SVM** with a **Convolutional Neural Network (CNN)**.

---

## 📌 Overview

Facial Emotion Recognition (FER) is a computer vision task that identifies human emotions from facial expressions.

This project uses the **FER2013 dataset** to classify facial expressions into seven emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

The project implements both **traditional machine learning** and **deep learning** approaches to study how different feature representations affect classification performance.

---

## 🎯 Objectives

* Perform exploratory data analysis on the FER2013 dataset.
* Preprocess and normalize facial images.
* Extract handcrafted features using **Histogram of Oriented Gradients (HOG)**.
* Apply **Principal Component Analysis (PCA)** for dimensionality reduction.
* Train an **SVM classifier** using HOG features.
* Train an **HOG + PCA + SVM** classification pipeline.
* Build a **CNN-based facial emotion classifier**.
* Evaluate models using standard classification metrics.
* Compare traditional machine learning with deep learning.

---

## 📊 Dataset

This project uses the **FER2013 (Facial Expression Recognition 2013)** dataset.

**Dataset:** FER2013
**Source:** Kaggle
**Link:** https://www.kaggle.com/datasets/msambare/fer2013

The dataset contains grayscale facial images with a resolution of **48 × 48 pixels**.

### Emotion Classes

| Label | Emotion  |
| :---: | -------- |
|   0   | Angry    |
|   1   | Disgust  |
|   2   | Fear     |
|   3   | Happy    |
|   4   | Sad      |
|   5   | Surprise |
|   6   | Neutral  |

---

## 🧠 Methodology

The complete workflow is:

```text
                 FER2013 Dataset
                        │
                        ▼
               Exploratory Data Analysis
                        │
                        ▼
                Image Preprocessing
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        HOG Feature            CNN Model
        Extraction
              │                   │
              ▼                   │
        Feature Scaling           │
              │                   │
              ▼                   │
             SVM                  │
              │                   │
              ▼                   │
             PCA                  │
              │                   │
              ▼                   │
             SVM                  │
              │                   │
              └─────────┬─────────┘
                        ▼
                Model Evaluation
                        │
                        ▼
                 Model Comparison
```

---

## 🔬 Techniques Used

### 1. Image Preprocessing

The images are:

* Converted to grayscale where required
* Reshaped to `48 × 48`
* Normalized to the range `[0, 1]`
* Converted into numerical arrays
* Encoded according to their emotion labels

The CNN receives images with the shape:

```text
48 × 48 × 1
```

---

### 2. HOG Feature Extraction

**Histogram of Oriented Gradients (HOG)** is used to extract handcrafted features from facial images.

HOG captures local edge and gradient information that can represent important facial structures and contours.

The traditional machine learning pipeline is:

```text
Image
  ↓
Preprocessing
  ↓
HOG Features
  ↓
Feature Scaling
  ↓
SVM
  ↓
Emotion Prediction
```

---

### 3. Support Vector Machine

A **Support Vector Machine (SVM)** classifier is trained using HOG features.

SVM is used as a traditional statistical pattern recognition classifier for distinguishing the seven emotion classes.

---

### 4. PCA Dimensionality Reduction

**Principal Component Analysis (PCA)** is applied to the extracted HOG features.

The pipeline is:

```text
Image
  ↓
HOG
  ↓
StandardScaler
  ↓
PCA
  ↓
SVM
  ↓
Emotion Prediction
```

PCA is used to investigate the effect of dimensionality reduction on:

* Number of features
* Computational complexity
* Information retention
* Classification performance

---

### 5. Convolutional Neural Network

A **Convolutional Neural Network (CNN)** is trained directly on the facial images.

The CNN automatically learns visual features from the input images.

The architecture includes:

* Convolutional layers
* Batch Normalization
* ReLU activation
* Max Pooling
* Dropout
* Dense layers
* Softmax output layer

The final layer produces probabilities for the seven emotion classes.

---

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

The project compares:

| Model           | Feature Representation | Dimensionality Reduction |
| --------------- | ---------------------- | ------------------------ |
| HOG + SVM       | HOG                    | None                     |
| HOG + PCA + SVM | HOG                    | PCA                      |
| CNN             | Learned CNN features   | None                     |

The actual performance values are generated by executing the notebook.

---

## 📓 Project Structure

This project intentionally uses a single Jupyter Notebook:

```text
Facial-Emotion-Recognition/
│
├── Facial_Emotion_Recognition.ipynb
├── README.md
└── .gitignore
```

All preprocessing, feature extraction, model training, evaluation, visualization, and comparison are implemented inside the notebook.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn opencv-python pillow scikit-learn scikit-image tensorflow jupyter
```

---

## 📥 Dataset Setup

Download the FER2013 dataset from Kaggle:

https://www.kaggle.com/datasets/msambare/fer2013

After downloading the dataset, update the dataset path inside the notebook if required.

> **Note:** The FER2013 dataset is not included in this repository because of its size and dataset licensing/distribution considerations.

---

## ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Facial_Emotion_Recognition.ipynb
```

Run the cells sequentially.

The notebook performs the following steps:

```text
1. Dataset Loading
2. Exploratory Data Analysis
3. Image Preprocessing
4. HOG Feature Extraction
5. HOG + SVM
6. HOG + PCA + SVM
7. CNN Training
8. Model Evaluation
9. Confusion Matrix
10. Sample Predictions
11. Model Comparison
12. Conclusion
```

---

## 📊 Results

The notebook generates the actual results after training and evaluation.

The final comparison includes:

| Model           | Accuracy | Precision | Recall | F1-Score |
| --------------- | -------: | --------: | -----: | -------: |
| HOG + SVM       |        — |         — |      — |        — |
| HOG + PCA + SVM |        — |         — |      — |        — |
| CNN             |        — |         — |      — |        — |

> Results should be updated with the actual values obtained after running the notebook. No performance values are assumed or fabricated.

---

## 🔍 Sample Predictions

The CNN model can be used to predict emotions for unseen facial images.

Example:

```text
Actual Emotion    : Happy
Predicted Emotion : Happy
Confidence        : XX.XX%
```

The prediction and confidence are generated directly from the trained model.

---

## 📚 Concepts Demonstrated

This project covers important concepts from **Statistical Pattern Recognition, Machine Learning, and Computer Vision**:

* Pattern Recognition
* Image Preprocessing
* Feature Extraction
* HOG
* Feature Scaling
* PCA
* Dimensionality Reduction
* Support Vector Machine
* Convolutional Neural Network
* Image Classification
* Model Evaluation
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Overfitting
* Generalization

---

## 🚀 Future Improvements

Possible future extensions include:

* Transfer learning using pretrained CNN architectures
* Real-time webcam-based emotion recognition
* Face detection before emotion classification
* Improved handling of class imbalance
* Advanced data augmentation
* Hyperparameter optimization
* Model compression and optimization
* Real-time emotion recognition applications

---

## ⚠️ Limitations

FER2013 contains variations in:

* Facial pose
* Lighting conditions
* Image quality
* Facial appearance
* Expression intensity

Some emotions can also have visually similar facial expressions, which may lead to misclassification.

Therefore, multiple evaluation metrics should be considered rather than relying only on accuracy.

---

## 👨‍💻 Author

**Yug Sondagar**

B.Tech — Computer Science / Artificial Intelligence & Machine Learning

---

## 📄 License

This project is developed for **educational and academic purposes**.
