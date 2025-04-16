import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Set page title
st.title("Prediction of Cardiovascular Risk in New–onset T2D")
st.caption("Based on TyG Index and Carotid Ultrasound Features")

# ===== Load model and data =====
model = joblib.load('LGB.pkl')  # Trained LightGBM model
X_test = pd.read_csv('x_test.csv')  # Original test set for SHAP/LIME context

# ===== Feature list (Displayed names) =====
feature_names = [
    "Age (years)",
    "Hypertension",
    "TyG index",  # Moved up here
    "IMT (mm)",  # Moved down here
    "Maximum plaque thickness (mm)",  # Updated here
    "Carotid plaque burden"  # Moved to the last position
]

# ===== Input form =====
with st.form("input_form"):
    st.subheader("Please enter the following clinical and ultrasound features:")
    inputs = []

    # Ensure features are entered in the correct order
    for col in feature_names:
        if col == "Hypertension":
            inputs.append(st.selectbox(col, options=[0, 1], index=0))

        elif col == "Age (years)":
            min_val = int(X_test["Age (years)"].min())
            max_val = 100
            default_val = int(X_test["Age (years)"].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=1)
            )

        elif col == "Maximum plaque thickness (mm)":
            min_val = 0.0
            max_val = 7.0
            default_val = float(X_test["Maximum plaque thickness (mm)"].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
            )

        elif col == "IMT (mm)":  # Handling IMT (mm)
            min_val = 0.5  # Adjust this min value based on your dataset's typical range
            max_val = 1.5  # Adjust this max value as per your dataset
            default_val = float(X_test["IMT (mm)"].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
            )

        elif col == "TyG index":  # Handling TyG index
            min_val = 1.0  # Adjust the minimum value based on dataset
            max_val = 15.0  # Adjust this max value as per your dataset
            default_val = float(X_test["TyG index"].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.01, format="%.2f")
            )

        elif col == "Carotid plaque burden":
            min_val = int(X_test[col].min())
            max_val = 15
            default_val = int(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=1)
            )

        else:
            min_val = float(X_test[col].min())
            max_val = float(X_test[col].max())
            default_val = float(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val)
            )

    submitted = st.form_submit_button("Submit Prediction")

# ===== Prediction and interpretation =====
if submitted:
    input_data = pd.DataFrame([inputs], columns=feature_names)
    input_data = input_data.round(2)  # Round to two decimal places for display
    st.subheader("Model Input Features")
    st.dataframe(input_data)

    # Prepare model input with original column names (adjusted for new feature names)
    model_input = pd.DataFrame([{
        "Age (years)": input_data["Age (years)"].iloc[0],
        "Hypertension": input_data["Hypertension"].iloc[0],
        "TyG index": input_data["TyG index"].iloc[0],
        "IMT (mm)": input_data["IMT (mm)"].iloc[0],
        "Maximum plaque thickness (mm)": input_data["Maximum plaque thickness (mm)"].iloc[0],
        "Carotid plaque burden": input_data["Carotid plaque burden"].iloc[0]
    }])

    predicted_proba = model.predict_proba(model_input)[0]
    probability = predicted_proba[1] * 100

    # ===== Risk Stratification by Percentile ===== 
    y_probs = model.predict_proba(X_test)[:, 1]
    low_threshold = np.percentile(y_probs, 50.0)  # 前50%
    mid_threshold = np.percentile(y_probs, 88.07)  # 前50% + 38.07% = 88.07%

    if predicted_proba[1] <= low_threshold:
        risk_level = "🟢 **You are currently at a low risk of cardiovascular disease.**"
        suggestion = "✅ Please continue to maintain a healthy lifestyle and attend regular follow-up visits."
    elif predicted_proba[1] <= mid_threshold:
        risk_level = "🟡 **You are at a moderate risk of cardiovascular disease.**"
        suggestion = "⚠️ It is advised to monitor your condition closely and consider preventive interventions."
    else:
        risk_level = "🔴 **You are at a high risk of cardiovascular disease.**"
        suggestion = "🚨 It is recommended to consult a physician promptly and take proactive medical measures."

    # ==== Display Result ==== 
    st.subheader("Prediction Result & Explanation")
    st.markdown(f"**Estimated probability:** {probability:.1f}%")
    st.info(risk_level)
    st.markdown(suggestion)

    # ===== SHAP Force Plot =====
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_input)

    if isinstance(shap_values, list):
        shap_value_sample = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_value_sample = shap_values
        expected_value = explainer.expected_value

    force_plot = shap.force_plot(
        base_value=expected_value,
        shap_values=shap_value_sample,
        features=model_input,
        matplotlib=True,
        show=False
    )

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    plt.close()
    st.image("shap_force_plot.png")

