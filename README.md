# 💳 Credit Card Fraud Detection (End-to-End ML + Deployment Project)

> A production-focused machine learning system for detecting fraudulent transactions, designed with real-world constraints like class imbalance, model reliability, and deployment in mind.

---

## 🚀 Project Overview

Fraud detection is a **high-impact, real-world ML problem** where mistakes are costly:

- ❌ Missing fraud = financial loss  
- ❌ Flagging normal transactions = poor user experience  

In this project, I built a **complete ML pipeline** from data exploration to deployment, focusing on:

- Handling **extreme class imbalance (~0.17% fraud)**
- Maximizing **fraud detection (recall)**
- Building **robust, scalable models**
- Avoiding **data leakage**
- Preparing for **real-world deployment (API + Streamlit)**

---

## 🎯 Problem Statement

Predict whether a transaction is:

- `0` → Legitimate  
- `1` → Fraud  

### ⚠️ Real Challenge:
> Build a model that **detects fraud effectively** while keeping false alarms low.

---

## 📊 Dataset

- Source: Kaggle (Credit Card Fraud Detection)
- Total samples: **284,807**
- Fraud cases: **492 (~0.17%)**
- Features:
  - `V1–V28` → PCA-transformed (anonymized)
  - `Time`, `Amount`
  - Target: `Class`

---

## ⚠️ Core Challenge: Extreme Imbalance

| Class | Count |
|------|------|
| Normal (0) | ~284,315 |
| Fraud (1)  | 492 |

👉 A model predicting all zeros gives **99.8% accuracy** — but is useless.

---

## 🧠 Full ML Workflow

1. Data loading & exploration  
2. Data preprocessing  
3. Train-test split (**before SMOTE**)  
4. Handling imbalance with **SMOTE**  
5. Model training (Logistic → RF → XGBoost → LightGBM)  
6. Evaluation using **F1, Recall, ROC-AUC**  
7. Hyperparameter tuning  
8. Model selection  
9. Model saving (`.pkl`)  
10. Deployment (Streamlit App / API)

---

## 🧪 Model Development Journey

---

### 🔹 Model 1: Logistic Regression (Baseline)

**Why used:**
- Simple, interpretable baseline

**Result:**
- Good accuracy but weak fraud detection

**Problem:**
- Missed too many fraud cases  
- Could not capture non-linear relationships  

> ❌ Dropped due to low recall on fraud

---

### 🔹 Critical Mistake: SMOTE Data Leakage

#### ❌ What went wrong:
Applied SMOTE **before train-test split**

- Caused **data leakage**
- Produced **unrealistically high performance**

#### ✅ Fix:
```python
# Correct workflow
X_train, X_test, y_train, y_test = train_test_split(...)

smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

#### 💡 Lesson:
> Always apply SMOTE **only on training data**

---

### 🌲 Model 2: Random Forest

**Why:**
- Handles non-linear patterns
- Robust and reliable baseline

**Improvement:**
- Better recall than Logistic Regression
- More stable predictions

---

### 🚀 Model 3: XGBoost

**Why:**
- High performance on tabular data
- Handles complex relationships
- Built-in regularization

**Impact:**
- Improved fraud detection
- Better ROC-AUC
- Strong generalization

---

### ⚡ Model 4: LightGBM

**Why:**
- Faster than XGBoost
- Efficient on large datasets
- Handles imbalance well

**Impact:**
- Comparable or better performance
- Faster training
- More scalable

---

## 📊 Final Model Performance (Best Model - XGBOOST)

| Metric        | Score |
|--------------|------|
| Accuracy     | ~99.95% |
| Precision    | High |
| Recall       | ~80% |
| F1 Score     | ~0.84 |
| ROC-AUC      | ~0.96 |

---

## 🔍 Confusion Matrix (Final)

|          | Pred 0 | Pred 1 |
|----------|--------|--------|
| Actual 0 | 56651  | 22     |
| Actual 1 | 9      | 73     |

---

## 🧠 Key Learnings

- Accuracy is **misleading** for imbalanced data  
- Recall & F1-score are more important  
- SMOTE must be applied carefully  
- Tree-based models outperform linear models here  
- Feature consistency is critical for deployment  

---

## 🔑 Feature Importance

Top predictors:

- `V14`
- `V12`
- `V17`

These features strongly influence fraud detection.

---

## ⚙️ Deployment

### 🖥️ Streamlit App
- Interactive UI for predictions  
- Dynamic feature input handling  
- Displays fraud probability  

### 🔌 API (FastAPI - optional)
- `/predict` endpoint  
- Accepts JSON transaction data  
- Returns prediction  

---

## 📁 Project Structure

```
fraud-detection-project/
│
├── data/
├── notebooks/
├── models/
│   └── fraud_model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── api/
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## ⚠️ Challenges Faced & Solutions

### ❌ 1. Class Imbalance
- Problem: Model ignored fraud cases  
- Solution: Used **SMOTE**  

---

### ❌ 2. Data Leakage
- Problem: Unrealistic performance  
- Solution: Applied SMOTE after split  

---

### ❌ 3. Feature Mismatch in Deployment
- Problem: Model errors during prediction  
- Solution: Used `feature_names_in_` for consistency  

---

### ❌ 4. Poor Model Performance (Initial)
- Problem: Logistic Regression underperformed  
- Solution: Switched to **RF → XGBoost → LightGBM**

---

### ❌ 5. Environment Issues (Streamlit/Packages)
- Problem: Commands not recognized  
- Solution: Used `python -m` execution method  

---

## 📈 Business Impact

This system can:

- Detect fraud **in real-time**
- Reduce financial losses  
- Improve customer trust  
- Minimize false alerts  

---

## 🏁 Final Takeaway

> Fraud detection is not about being right most of the time —  
> it's about **catching the rare, costly mistakes** without disrupting normal users.

---

## 👨‍💻 Author

Built with a strong focus on:

- Real-world ML workflows  
- Problem-solving mindset  
- Production-ready thinking  
- Continuous learning  

---
