import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 设置页面标题
st.title("Prediction of Cardiovascular Risk in New–onset T2D")
st.caption("Based on TyG Index and Carotid Ultrasound Features")

# ===== 加载模型和数据 =====
model = joblib.load('LGB.pkl')  # 已训练的 LightGBM 模型
X_test = pd.read_csv('x_test.csv')  # 用于 SHAP/LIME 解释的数据集

# ===== 特征名称（显示名称） =====
feature_names = [
    "Age (years)",
    "Hypertension",
    "TyG index",  # 将其放在这里
    "IMT (mm)",  # 将其放在这里
    "Maximum plaque thickness (mm)",  # 更新这里
    "Carotid plaque burden"  # 放在最后
]

# ===== 输入表单 =====
with st.form("input_form"):
    st.subheader("Please enter the following clinical and ultrasound features:")
    inputs = []

    # 确保按正确的顺序输入特征
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

        elif col == "IMT (mm)":  # 处理 IMT (mm)
            min_val = 0.0  # 设置最小值为 0.00
            max_val = 1.5
            # Ensure no missing values in the column
            default_val_IMT = X_test["IMT (mm)"].dropna().median() if not X_test["IMT (mm)"].isnull().all() else 0.0
            inputs.append(
                st.number_input(col, value=default_val_IMT, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
            )

        elif col == "TyG index":  # 处理 TyG index
            min_val = 0.0  # 设置最小值为 0.00
            max_val = 15.0
            # Ensure no missing values in the column
            default_val_tyG = X_test["TyG index"].dropna().median() if not X_test["TyG index"].isnull().all() else 0.0
            inputs.append(
                st.number_input(col, value=default_val_tyG, min_value=min_val, max_value=max_val, step=0.01, format="%.2f")
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

# ===== 预测与解释 =====
if submitted:
    input_data = pd.DataFrame([inputs], columns=feature_names)
    input_data = input_data.round(2)  # 将输入四舍五入到两位小数以供显示
    st.subheader("Model Input Features")
    st.dataframe(input_data)

    # 准备模型输入（使用原始列名，调整为新的特征名称）
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

    # ===== 风险分层（按百分位） ===== 
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

    # ==== 显示结果 ==== 
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


