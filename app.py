import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import glob

# Configure matplotlib for better visualization
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10

# Set page config with custom icon
st.set_page_config(
    page_title="Model prediction visualization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}
    .stNumberInput>label {font-weight: bold; color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #e9ecef;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("Model prediction visualization")
st.markdown("""
    This tool uses feature data to make predictions and provides mechanistic explanations through SHAP visualization.
    Adjust the feature value in the sidebar to observe the changes in the prediction results and SHAP values.
""")


# Find the Excel file in the data folder
def get_excel_file():
    excel_files = glob.glob('data/*.xlsx')
    if len(excel_files) != 1:
        st.error("Expected exactly one Excel file in the 'data' folder.")
        return None
    return excel_files[0]


# Load and prepare background data
@st.cache_data
def load_background_data():
    excel_file = get_excel_file()
    if excel_file is None:
        return None
    df = pd.read_excel(excel_file)
    return df.iloc[:, :-2]  # Exclude the last column (target)


# Determine model type and class labels
@st.cache_data
def determine_model_type():
    excel_file = get_excel_file()
    if excel_file is None:
        return None, None
    df = pd.read_excel(excel_file)
    target_col = df.iloc[:, -1]  # Get the last column (target)
    unique_values = target_col[1:].nunique()  # Exclude header, count unique values
    if unique_values == 2:
        model_type = "classification"
        labels = list(target_col[1:].unique())  # Get the two unique labels
        return model_type, labels
    else:
        model_type = "regression"
        return model_type, None


# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/MODEL.h5')


# Initialize data and model
background_data = load_background_data()
if background_data is None:
    st.stop()

model = load_model()

# Default values for features
default_values = background_data.iloc[0, :].to_dict()

# Determine model type
model_type, class_labels = determine_model_type()
if model_type is None:
    st.stop()

# Sidebar configuration
st.sidebar.header("Feature Inputs")
st.sidebar.markdown("Adjust values of features:")

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.update(default_values)

features = list(default_values.keys())
values = {}
cols = st.sidebar.columns(2)

for i, feature in enumerate(features):
    with cols[i % 2]:
        values[feature] = st.number_input(
            feature,
            min_value=float(background_data[feature].min()),
            max_value=float(background_data[feature].max()),
            value=default_values[feature],
            step=0.001,
            format="%.3f",
            key=feature
        )


# Prepare input data
def prepare_input_data():
    return pd.DataFrame([values])


# Main analysis
if st.button("Analyze Calculation", key="calculate"):
    input_df = prepare_input_data()

    # Prediction
    prediction = model.predict(input_df.values, verbose=0)[0][0]
    with st.container():
        st.header("📈 Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            if model_type == "classification":
                # Classification model display
                predicted_class = class_labels[1] if prediction >= 0.5 else class_labels[0]
                st.metric(
                    "Probability",
                    f"{prediction:.4f}",
                    delta=f"Predicted Class: {predicted_class}",
                    delta_color="inverse"
                )
            else:
                # Regression model display
                st.metric(
                    "Predicted Value",
                    f"{prediction:.4f}"
                )
        with col2:
            if model_type == "classification":
                st.metric(
                    "Classification Threshold",
                    f"0.5"
                )
            else:
                # Display statistical information for regression model
                excel_file = get_excel_file()
                df_y = pd.read_excel(excel_file).iloc[:, -1]
                st.metric(
                    "Prediction Range",
                    f"{df_y.min():.2f} - {df_y.max():.2f}"
                )

    # SHAP explanation
    explainer = shap.DeepExplainer(model, background_data.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())

    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["Force Plot", "Decision Plot", "Mechanistic Insights"])

    with tab1:
        st.subheader("Force Plot")
        col1, col2 = st.columns([3, 1])
        with col1:
            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                feature_names=input_df.columns,
                data=input_df.values.round(3)
            )
            shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 4))
            st.pyplot(plt.gcf(), clear_figure=True)

    with tab2:
        st.subheader("Decision Plot")
        col1, col2 = st.columns([2, 2])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            shap.decision_plot(base_value, shap_values, input_df.columns, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)

    with tab3:
        st.subheader("Mechanistic Insights")
        importance_df = pd.DataFrame({'Feature': input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))