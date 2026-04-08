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

| Metric        | Class 0 | Class 1 |
|--------------|--------|--------|
| Precision    | 1.00   | 0.83   |
| Recall       | 1.00   | 0.86   |
| F1-score     | 1.00   | 0.84   |

- **ROC-AUC:** 0.9852  
- **Accuracy:** ~1.00  

---

### 🔍 Confusion Matrix

|          | Pred 0 | Pred 1 |
|----------|--------|--------|
| Actual 0 | 56847  | 17     |
| Actual 1 | 14     | 84     |

---

### 🧠 Key Insight

- Best-performing model across all experiments  
- Achieved the **highest ROC-AUC (~0.985)**  
- Strong balance between **precision and recall**  
- Successfully reduced both:
  - False Negatives (missed fraud)
  - False Positives (false alarms)

> 🏆 **Selected as the final production model**
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

| Metric        | Class 0 | Class 1 |
|--------------|--------|--------|
| Precision    | 1.00   | 0.84   |
| Recall       | 1.00   | 0.85   |
| F1-score     | 1.00   | 0.84   |

- **ROC-AUC:** 0.9843  
- **Accuracy:** ~1.00  

---

### 🔍 Confusion Matrix

|          | Pred 0 | Pred 1 |
|----------|--------|--------|
| Actual 0 | 56848  | 16     |
| Actual 1 | 15     | 83     |

---

### 🧠 Key Insight

- Performance very close to XGBoost  
- Slightly lower ROC-AUC (**0.984 vs 0.985**)  
- Maintains strong balance between **precision and recall**  
- Efficient and faster to train compared to XGBoost  

> ⚡ **Excellent alternative model with high performance and scalability**

- Performance similar to XGBoost  
- Faster training time  
- Slightly lower consistency in fraud recall  

### ✅ Insight
- Great for scalability
- XGBoost still slightly better for this dataset

---

# 📊 Model Comparison Summary

To evaluate performance across models, the focus was placed on **fraud detection capability (Recall, F1-score, ROC-AUC)** rather than accuracy.

---

## 🧪 Model Performance Comparison

| Model                | ROC-AUC | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) | Notes |
|---------------------|--------|------------------|---------------|------------------|------|
| Logistic Regression | 0.9522 | 0.8889           | 0.7579        | 0.8182           | Baseline model, struggled with non-linearity |
| Random Forest       | ~0.96  | 0.8837           | 0.8000        | 0.8398           | Strong improvement, stable performance |
| XGBoost (Advanced)  | **0.9852** | 0.83         | **0.86**      | **0.84**         | 🏆 Best overall model |
| LightGBM (Advanced) | 0.9843 | **0.84**         | 0.85          | 0.84             | Fast and highly competitive |

---

## 🔍 Key Observations

- **XGBoost achieved the highest ROC-AUC (~0.985)** → best overall separation of classes  
- **LightGBM closely matched performance** with slightly faster training  
- **Random Forest improved recall significantly** over Logistic Regression  
- **Logistic Regression underperformed** due to inability to capture complex patterns  

---

## 🧠 Final Model Selection

> 🏆 **XGBoost was selected as the final model** because it provided:

- Best **overall performance (ROC-AUC)**
- Strong **fraud detection (Recall = 0.86)**
- Balanced **precision-recall tradeoff**
- Robust performance across all evaluation metrics  

---

## ⚖️ Trade-Off Insight

- Increasing recall (catching more fraud) often reduces precision (more false alarms)  
- The final model strikes a **practical balance suitable for real-world deployment**

---

## 💡 Key Takeaway

> The best model is not the most accurate —  
> it’s the one that **catches the most fraud without overwhelming the system with false positives**

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
