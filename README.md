# 💳 Credit Card Fraud Detection (End-to-End ML + Deployment Project)

> A production-focused machine learning system for detecting fraudulent transactions, designed with real-world constraints like class imbalance, model reliability, and deployment in mind.

---

## 🚀 Project Overview

Fraud detection is a **high-impact ML problem** where mistakes are costly:

- ❌ Missing fraud → financial loss  
- ❌ False alarms → poor user experience  

This project builds a **complete ML pipeline**, focusing on:

- Handling **extreme class imbalance (~0.17%)**
- Maximizing **fraud detection (recall)**
- Comparing multiple models
- Avoiding **data leakage**
- Preparing for **real-world deployment**

---

## 🎯 Problem Statement

Predict:

- `0` → Legitimate  
- `1` → Fraud  

> Goal: Detect fraud **without overwhelming users with false positives**

---

## 📊 Dataset

- **284,807 transactions**
- **492 fraud cases (~0.17%)**
- Features:
  - `V1–V28` (PCA transformed)
  - `Time`, `Amount`
  - Target: `Class`

---

## ⚠️ Core Challenge: Imbalanced Data

| Class | Count |
|------|------|
| Normal | ~284,315 |
| Fraud  | 492 |

👉 Accuracy alone is useless here.

---

## 🧠 Workflow

1. Data Exploration  
2. Preprocessing  
3. Train/Test Split  
4. SMOTE (train only)  
5. Model Training  
6. Evaluation (F1, Recall, ROC-AUC)  
7. Model Comparison  
8. Threshold Tuning  
9. Deployment  

---

# 🧪 Model Results & Evolution

---

## 🔹 1. Logistic Regression (Baseline)

### 📊 Results

| Metric        | Class 0 | Class 1 |
|--------------|--------|--------|
| Precision    | 0.999594 | 0.888889 |
| Recall       | 0.999841 | 0.757895 |
| F1-score     | 0.999718 | 0.818182 |

- **ROC-AUC:** 0.9522  
- **Accuracy:** 0.9994  

### ❌ Issues
- Missed **fraud cases (low recall)**
- Could not capture **non-linear patterns**

> ❌ Dropped

---

## ⚠️ Critical Mistake: SMOTE Data Leakage

### ❌ Problem
- Applied SMOTE **before splitting**
- Result: Unrealistically high performance

### ✅ Fix
- Applied SMOTE **only on training data**

> 💡 Key Lesson: Avoid data leakage at all costs

---

## 🌲 2. Random Forest

### 📊 Initial Results

| Metric        | Class 0 | Class 1 |
|--------------|--------|--------|
| Precision    | 0.999612 | 0.890244 |
| Recall       | 0.999841 | 0.768421 |
| F1-score     | 0.999726 | 0.824859 |

- **ROC-AUC:** 0.9578  

---

### 🚀 Optimized Random Forest

| Metric        | Class 0 | Class 1 |
|--------------|--------|--------|
| Precision    | 0.999665 | 0.883721 |
| Recall       | 0.999823 | 0.800000 |
| F1-score     | 0.999744 | 0.839779 |

- **Accuracy:** 0.99949  
- **ROC-AUC:** ~0.96  

### ✅ Improvements
- Better fraud recall
- More stable than Logistic Regression

---

## 🚀 3. XGBoost (Best Model)

### 📊 Results
=== ADVANCED XGBOOST RESULTS ===
ROC-AUC SCORE: 0.9852393968279489
CLASSIFICATION REPORT:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.83      0.86      0.84        98

    accuracy                           1.00     56962
   macro avg       0.92      0.93      0.92     56962
weighted avg       1.00      1.00      1.00     56962

CONFUSION MATRIX:
 [[56847    17]
 [   14    84]]

- **Accuracy:** ~99.95%  
- **ROC-AUC:** ~0.97  
- **F1-score (Fraud):** ~84% Highest among all models  
- **Recall (Fraud):** ~86% Best balance achieved  

### ✅ Why XGBoost Won

- Captures complex patterns
- Handles imbalance well
- Regularization reduces overfitting
- Strong generalization

> 🏆 **Selected as Final Model**

---

## ⚡ 4. LightGBM

### 📊 Results

=== ADVANCED LIGHTGBM RESULTS ===
ROC-AUC SCORE: 0.9843001705465528
CLASSIFICATION REPORT:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.84      0.85      0.84        98

    accuracy                           1.00     56962
   macro avg       0.92      0.92      0.92     56962
weighted avg       1.00      1.00      1.00     56962

CONFUSION MATRIX:
 [[56848    16]
 [   15    83]]

- Performance similar to XGBoost  
- Faster training time  
- Slightly lower consistency in fraud recall  

### ✅ Insight
- Great for scalability
- XGBoost still slightly better for this dataset

---

# 🎯 Threshold Tuning Experiment

### 🔍 Goal
Improve fraud detection by adjusting prediction threshold (instead of default 0.5)

---

### ⚠️ What Happened

- Lower threshold → higher recall  
- But also → massive increase in false positives  

---

### 📊 Outcome

| Threshold | Recall ↑ | Precision ↓ |
|----------|--------|------------|
| Lowered  | Improved | Dropped significantly |

---

### ❌ Final Decision

> Threshold tuning was **not used in final model** because:

- Too many false alarms
- Bad user experience
- Not practical in real-world deployment

---

# 🔍 Final Confusion Matrix (Best Model)

|          | Pred 0 | Pred 1 |
|----------|--------|--------|
| Actual 0 | 56651  | 22     |
| Actual 1 | 9      | 73     |

---

# 🧠 Key Insights

- Accuracy is misleading in imbalanced data  
- Recall is critical for fraud detection  
- SMOTE must be applied correctly  
- Tree-based models outperform linear models  
- Threshold tuning must consider business impact  

---

# 🔑 Feature Importance

Top features:

- `V14`
- `V12`
- `V17`

---

# ⚙️ Deployment

### 🖥️ Streamlit App
- User-friendly fraud prediction UI  
- Real-time predictions  
- Displays probability  

### 🔌 API (Optional)
- `/predict` endpoint  
- JSON input/output  

---

# 📁 Project Structure

```
fraud-detection-project/
│
├── data/
├── notebooks/
├── models/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

# ⚠️ Challenges & Solutions

### ❌ Class Imbalance
→ ✅ Solved with SMOTE  

### ❌ Data Leakage
→ ✅ Fixed by correct pipeline order  

### ❌ Weak Baseline Model
→ ✅ Switched to RF → XGBoost → LightGBM → XGBoost 

### ❌ Threshold Tradeoff
→ ✅ Balanced using default threshold  

### ❌ Deployment Issues
→ ✅ Fixed feature mismatch + environment setup  

---

# 📈 Business Impact

- Detect fraud in real-time  
- Reduce financial losses  
- Improve trust  
- Maintain good user experience  

---

# 🏁 Final Takeaway

> Fraud detection is not about accuracy —  
> it's about **catching rare, high-impact events without hurting normal users**

---

## 👨‍💻 Author
**Tobychuks**

Focused on:

- Real-world ML workflows  
- Problem-solving mindset  
- Production-ready systems
- catching fraud while keeping the balance between precision and recall 

---
