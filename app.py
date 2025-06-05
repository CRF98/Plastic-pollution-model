import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import glob
import json

# 配置 matplotlib
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10

# 设置页面配置
st.set_page_config(
    page_title="模型预测可视化",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
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

# 标题和简介
st.title("模型预测可视化")
st.markdown("""
    本工具使用特征数据进行预测，并通过 SHAP 可视化提供机理解释。
    在侧边栏调整特征值，观察预测结果和 SHAP 值的变化。
""")

# 查找 Excel 文件
def get_excel_file():
    excel_files = glob.glob('data/*.xlsx')
    if len(excel_files) != 1:
        st.error("data 文件夹中应有且仅有一个 Excel 文件。")
        return None
    return excel_files[0]

# 查找 JSON 文件
def get_json_file():
    json_files = glob.glob('data/*.json')
    if len(json_files) != 1:
        st.error("data 文件夹中应有且仅有一个 JSON 文件。")
        return None
    return json_files[0]

# 加载归一化方法
@st.cache_data
def load_norm_method():
    json_file = get_json_file()
    if json_file is None:
        return 'none'
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
        norm_method = config.get('input_norm', 'none')
        if norm_method not in ['ext', 'avg', 'avgext', 'avgstd', 'none']:
            st.error(f"JSON 文件中的归一化方法无效：{norm_method}。使用 'none'。")
            return 'none'
        return norm_method
    except Exception as e:
        st.error(f"读取 JSON 文件出错：{str(e)}。使用 'none'。")
        return 'none'

# 加载背景数据和归一化参数
@st.cache_data
def load_background_data():
    excel_file = get_excel_file()
    if excel_file is None:
        return None, None
    df = pd.read_excel(excel_file)
    features = df.iloc[:, :-2]  # 排除最后2列（目标列）
    param = {
        'mean': features.mean(),
        'max': features.max(),
        'std': features.std()
    }
    return features, param

# 确定模型类型和分类标签
@st.cache_data
def determine_model_type():
    excel_file = get_excel_file()
    if excel_file is None:
        return None, None
    df = pd.read_excel(excel_file)
    target_col = df.iloc[:, -1]  # 获取目标列
    unique_values = target_col[1:].nunique()  # 排除表头，统计唯一值
    if unique_values == 2:
        model_type = "classification"
        labels = list(target_col[1:].unique())  # 获取两个唯一标签
        return model_type, labels
    else:
        model_type = "regression"
        return model_type, None

# 加载预训练模型
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/MODEL.h5')

# 归一化操作
norm_op_dict = {
    'ext': lambda data, param: data / param.get('max', 1),
    'avg': lambda data, param: data - param.get('mean', 0),
    'avgext': lambda data, param: (data - param.get('mean', 0)) / param.get('max', 1),
    'avgstd': lambda data, param: (data - param.get('mean', 0)) / param.get('std', 1),
    'none': lambda data, param: data
}

# 反归一化操作
denorm_op_dict = {
    'ext': lambda data, param: data * param.get('max', 1),
    'avg': lambda data, param: data + param.get('mean', 0),
    'avgext': lambda data, param: data * param.get('max', 1) + param.get('mean', 0),
    'avgstd': lambda data, param: data * param.get('std', 1) + param.get('mean', 0),
    'none': lambda data, param: data
}

# 初始化数据和模型
background_data, norm_params = load_background_data()
if background_data is None:
    st.stop()

model = load_model()
norm_method = load_norm_method()

# 默认特征值（使用原始未归一化值）
default_values = background_data.iloc[0, :].to_dict()

# 确定模型类型
model_type, class_labels = determine_model_type()
if model_type is None:
    st.stop()

# 侧边栏配置
st.sidebar.header("特征输入")
st.sidebar.markdown("调整特征值：")

# 重置按钮
if st.sidebar.button("重置为默认值", key="reset"):
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

# 准备输入数据（归一化）
def prepare_input_data():
    input_df = pd.DataFrame([values])
    norm_func = norm_op_dict.get(norm_method, norm_op_dict['none'])
    normalized_data = input_df.copy()
    for col in input_df.columns:
        normalized_data[col] = norm_func(input_df[col], {
            'mean': norm_params['mean'].get(col, 0),
            'max': norm_params['max'].get(col, 1),
            'std': norm_params['std'].get(col, 1)
        })
    return normalized_data, input_df

# 主分析
if st.button("分析计算", key="calculate"):
    normalized_input_df, original_input_df = prepare_input_data()

    # 预测
    prediction = model.predict(normalized_input_df.values, verbose=0)[0][0]

    # 对于回归模型，反归一化预测值
    if model_type == "regression":
        denorm_func = denorm_op_dict.get(norm_method, denorm_op_dict['none'])
        excel_file = get_excel_file()
        df_y = pd.read_excel(excel_file).iloc[:, -1]  # 目标列
        target_params = {
            'mean': df_y.mean(),
            'max': df_y.max(),
            'std': df_y.std()
        }
        display_prediction = denorm_func(prediction, target_params)
    else:
        display_prediction = prediction  # 分类模型：使用原始概率

    # 显示预测结果
    with st.container():
        st.header("📈 预测结果")
        col1, col2 = st.columns(2)
        with col1:
            if model_type == "classification":
                predicted_class = class_labels[1] if prediction >= 0.5 else class_labels[0]
                st.metric(
                    "概率",
                    f"{prediction:.4f}",
                    delta=f"预测类别: {predicted_class}",
                    delta_color="inverse"
                )
            else:
                st.metric(
                    "预测值",
                    f"{display_prediction:.4f}"
                )
        with col2:
            if model_type == "classification":
                st.metric(
                    "分类阈值",
                    f"0.5"
                )
            else:
                excel_file = get_excel_file()
                df_y = pd.read_excel(excel_file).iloc[:, -1]
                st.metric(
                    "预测范围",
                    f"{df_y.min():.2f} - {df_y.max():.2f}"
                )

    # SHAP 解释
    explainer = shap.DeepExplainer(model, background_data.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(normalized_input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())

    # 可视化标签页
    tab1, tab2, tab3 = st.tabs(["Force Plot", "Decision Plot", "机理洞察"])

    with tab1:
        st.subheader("Force Plot")
        col1, col2 = st.columns([3, 1])
        with col1:
            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                feature_names=original_input_df.columns,
                data=original_input_df.values.round(3)
            )
            shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 4))
            st.pyplot(plt.gcf(), clear_figure=True)

    with tab2:
        st.subheader("Decision Plot")
        col1, col2 = st.columns([2, 2])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            shap.decision_plot(base_value, shap_values, original_input_df.columns, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)

    with tab3:
        st.subheader("机理洞察")
        importance_df = pd.DataFrame({'Feature': original_input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))