# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------
# 1️⃣ Load Model
# -------------------------
MODEL_PATH = os.path.join("models", "fraud_model.pkl")

@st.cache_data
def load_model():
    model = joblib.load(MODEL_PATH)
    # Get feature names to dynamically create input fields
    if hasattr(model, "get_booster"):  # XGBoost
        feature_names = model.get_booster().feature_names
    else:  # scikit-learn
        feature_names = model.feature_names_in_
    return model, feature_names

fraud_model, feature_names = load_model()

# -------------------------
# 2️⃣ Streamlit UI
# -------------------------
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below to predict if it's fraudulent.")

with st.form("transaction_form"):
    input_data = {}
    for feature in feature_names:
        default_val = 0.0 if feature != "Amount" else 100.0
        input_data[feature] = st.number_input(feature, min_value=0.0, value=default_val)

    submitted = st.form_submit_button("Predict Fraud")

# -------------------------
# 3️⃣ Make Prediction
# -------------------------
if submitted:
    try:
        # Convert input to DataFrame and reorder columns exactly as model expects
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]

        # Predict
        prediction = fraud_model.predict(input_df)[0]
        probability = fraud_model.predict_proba(input_df)[0][1]

        # Display result
        if prediction == 1:
            st.error(f"⚠️ Fraud Detected! Probability: {probability:.2f}")
        else:
            st.success(f"✅ Transaction Looks Safe. Probability of Fraud: {probability:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")