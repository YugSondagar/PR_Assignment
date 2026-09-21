# Facial Emotion Recognition Using FER2013

A machine learning and deep learning project for recognizing human facial emotions from facial images using the **FER2013 dataset**. The project explores traditional statistical pattern recognition techniques such as **HOG, PCA, and SVM**, and compares them with a **Convolutional Neural Network (CNN)**.

---

## 📌 Project Overview

Facial Emotion Recognition (FER) is a computer vision task that aims to identify human emotions from facial expressions.

In this project, facial images from the **FER2013 dataset** are processed and classified into seven emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

The project implements both **traditional machine learning** and **deep learning** approaches to understand how different feature representations affect classification performance.

---

## 🎯 Objectives

* Perform exploratory data analysis on the FER2013 dataset.
* Preprocess and normalize facial images.
* Extract handcrafted features using **Histogram of Oriented Gradients (HOG)**.
* Perform dimensionality reduction using **Principal Component Analysis (PCA)**.
* Build an **SVM classifier** using HOG features.
* Build an **HOG + PCA + SVM** classification pipeline.
* Develop a **CNN-based emotion classification model**.
* Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
* Compare traditional machine learning approaches with CNN-based feature learning.

---

## 📊 Dataset

This project uses the **FER2013 (Facial Expression Recognition 2013)** dataset.

**Dataset:** FER2013

**Source:** Kaggle
https://www.kaggle.com/datasets/msambare/fer2013

The images are grayscale facial images with a resolution of **48 × 48 pixels**.

### Emotion Classes

| Label | Emotion  |
| ----- | -------- |
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Sad      |
| 5     | Surprise |
| 6     | Neutral  |

---

## 🧠 Methodology

The project follows the pipeline:

```text
FER2013 Dataset
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Image Preprocessing
       │
       ├───────────────┐
       ▼               ▼
     HOG              CNN
   Features          Model
       │               │
       ▼               │
      SVM              │
       │               │
       ├──► PCA ──► SVM│
       │               │
       └───────────────┘
               │
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
* Converted into appropriate numerical representations

---

### 2. HOG Feature Extraction

**Histogram of Oriented Gradients (HOG)** is used to extract structural features from facial images.

HOG captures local edge and gradient information, which can represent important facial structures such as:

* Eyes
* Eyebrows
* Nose
* Mouth
* Facial contours

The extracted HOG features are then provided to traditional machine learning classifiers.

---

### 3. SVM Classification

A **Support Vector Machine (SVM)** classifier is trained using HOG features.

Pipeline:

```text
Image → HOG → Feature Scaling → SVM → Emotion
```

---

### 4. PCA Dimensionality Reduction

**Principal Component Analysis (PCA)** is applied to reduce the dimensionality of HOG features.

Pipeline:

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
Emotion
```

PCA helps investigate how dimensionality reduction affects:

* Number of features
* Computational requirements
* Classification performance
* Information retention

---

### 5. CNN

A Convolutional Neural Network is implemented to automatically learn useful visual features directly from the facial images.

The CNN includes layers such as:

* Convolution
* Batch Normalization
* ReLU activation
* Max Pooling
* Dropout
* Dense layers
* Softmax output

Input:

```text
48 × 48 × 1
```

Output:

```text
7 emotion classes
```

---

## 📈 Model Evaluation

The implemented models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

The notebook also compares the performance of:

```text
HOG + SVM
HOG + PCA + SVM
CNN
```

Actual evaluation values are generated after training and testing the models.

---

## 🗂️ Repository Structure

```text
Facial-Emotion-Recognition/
│
├── Facial_Emotion_Recognition.ipynb
├── README.md
└── .gitignore
```

The complete implementation is contained in the Jupyter Notebook.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Facial-Emotion-Recognition.git
```

Navigate to the project:

```bash
cd Facial-Emotion-Recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn opencv-python pillow scikit-learn scikit-image tensorflow jupyter
```

---

## 📥 Dataset Setup

Download the FER2013 dataset from Kaggle:

https://www.kaggle.com/datasets/msambare/fer2013

Place the dataset in the location expected by the notebook.

The notebook should be updated with the local dataset path before execution if necessary.

**Do not upload the complete FER2013 dataset to this repository.**

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

Run the notebook cells sequentially.

The notebook performs:

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

The notebook generates the actual results after training.

The main comparison includes:

| Model           | Feature Representation | Dimensionality Reduction |              Accuracy |              F1-Score |
| --------------- | ---------------------- | ------------------------ | --------------------: | --------------------: |
| HOG + SVM       | HOG                    | No                       | Generated by notebook | Generated by notebook |
| HOG + PCA + SVM | HOG                    | PCA                      | Generated by notebook | Generated by notebook |
| CNN             | Learned features       | No                       | Generated by notebook | Generated by notebook |

Results should be reported from the actual execution of the notebook rather than using predefined or assumed values.

---

## 🔍 Sample Prediction

The CNN can be used to predict the emotion of unseen facial images.

Example output:

```text
Actual Emotion: Happy
Predicted Emotion: Happy
Confidence: XX.XX%
```

The displayed confidence and prediction are generated directly from the trained model.

---

## 📚 Key Concepts Demonstrated

This project demonstrates several important concepts in **Statistical Pattern Recognition and Machine Learning**:

* Pattern recognition
* Image preprocessing
* Feature extraction
* HOG
* Dimensionality reduction
* PCA
* Feature scaling
* Support Vector Machines
* Convolutional Neural Networks
* Classification
* Model evaluation
* Confusion matrix
* Precision
* Recall
* F1-score
* Overfitting and generalization

---

## 🚀 Future Improvements

Possible extensions include:

* Transfer learning using pretrained CNN architectures
* Real-time webcam-based emotion recognition
* Face detection before emotion classification
* Improved handling of class imbalance
* Data augmentation
* Hyperparameter optimization
* Model compression for deployment
* Real-time emotion recognition applications

---

## ⚠️ Limitations

FER2013 contains variations in:

* Facial pose
* Lighting
* Image quality
* Facial appearance
* Expression intensity

Some emotions can also have visually similar facial expressions, which can lead to misclassification.

Therefore, model performance should be interpreted using multiple evaluation metrics rather than accuracy alone.

---

## 👨‍💻 Author

**Yug Sondagar**

B.Tech — Computer Science / Artificial Intelligence & Machine Learning

---

## 📄 License

This project is intended for **educational and academic purposes**.
#   P R _ A s s i g n m e n t  
 