"""
Pakistani Diabetes Risk Analysis — Interactive Streamlit App
Built from DataVisualizationProject4.ipynb

Run locally with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, classification_report
)

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Pakistani Diabetes Risk Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAL = {"Non-Diabetic": "#2ecc71", "Diabetic": "#e74c3c"}
C0, C1 = "#2ecc71", "#e74c3c"

RAW_RENAME_MAP = {
    'Age': 'Age', 'Gender': 'Gender', 'Region': 'Rgn', 'Rgn': 'Rgn',
    'Wt': 'Weight', 'Wt.': 'Weight', 'Wst': 'Waist',
    'sys': 'Systolic_BP', 'Sys': 'Systolic_BP',
    'dia': 'Diastolic_BP', 'Dia': 'Diastolic_BP',
    'his': 'Family_History', 'His': 'Family_History',
    'Alc': 'A1c', 'HbA1c': 'A1c',
    'B.S.#': 'Blood_Sugar', 'B.S': 'Blood_Sugar', 'B_S': 'Blood_Sugar', 'Blood_Sugar': 'Blood_Sugar',
    'vision': 'Vision_Problems', 'Vision': 'Vision_Problems',
    'Exer': 'Exr', 'Exr': 'Exr', 'Exercise_Min': 'Exr',
    'dipsia': 'Polydipsia', 'Dipsia': 'Polydipsia',
    'uria': 'Polyuria', 'Uria': 'Polyuria',
    'Dur': 'Duration', 'Duration': 'Duration',
    'neph': 'Nephropathy', 'Neph': 'Nephropathy',
    'HDL': 'HDL_Cholesterol', 'Outcome': 'Outcome', 'BMI': 'BMI',
    'Weight': 'Weight', 'Waist': 'Waist',
    'Systolic_BP': 'Systolic_BP', 'Diastolic_BP': 'Diastolic_BP',
    'Family_History': 'Family_History', 'A1c': 'A1c',
    'Vision_Problems': 'Vision_Problems', 'Polydipsia': 'Polydipsia',
    'Polyuria': 'Polyuria', 'Nephropathy': 'Nephropathy',
    'HDL_Cholesterol': 'HDL_Cholesterol',
}

LEAKAGE_COLS = [
    'Outcome', 'Outcome_Label', 'Age_Group', 'BMI_Category',
    'Duration', 'A1c', 'Polydipsia', 'Polyuria', 'B.S.R',
]

ZERO_CHECK_CANDIDATES = ['BMI', 'Blood_Sugar', 'A1c', 'Systolic_BP', 'Diastolic_BP', 'Weight', 'HDL_Cholesterol']

FEATURE_LABELS = {
    'Age': 'Age (years)', 'Gender': 'Gender (0=Female, 1=Male)',
    'Rgn': 'Region (0=Urban, 1=Rural)', 'Weight': 'Weight (kg)',
    'BMI': 'BMI', 'Waist': 'Waist (cm)', 'Systolic_BP': 'Systolic BP (mmHg)',
    'Diastolic_BP': 'Diastolic BP (mmHg)', 'Family_History': 'Family History (0=No, 1=Yes)',
    'Blood_Sugar': 'Blood Sugar (mg/dL)', 'Vision_Problems': 'Vision Problems (0=No, 1=Yes)',
    'Exr': 'Exercise (min/week)', 'Nephropathy': 'Nephropathy (0=No, 1=Yes)',
    'HDL_Cholesterol': 'HDL Cholesterol (mg/dL)',
}


# ------------------------------------------------------------------
# DATA LOADING & CLEANING (mirrors the notebook)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes_or_path):
    df = pd.read_csv(file_bytes_or_path)
    df.columns = df.columns.str.strip()

    rename_map = {k: v for k, v in RAW_RENAME_MAP.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    if 'Duration' in df.columns:
        df.drop(columns=['Duration'], inplace=True)

    zero_cols = [c for c in ZERO_CHECK_CANDIDATES if c in df.columns]
    for col in zero_cols:
        if (df[col] == 0).sum() > 0:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    df['Outcome_Label'] = df['Outcome'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100],
                              labels=['<30', '30-40', '40-50', '50-60', '60+'])
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                                 labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    return df


@st.cache_resource(show_spinner=False)
def train_all_models(df):
    feature_cols = [c for c in df.columns if c not in LEAKAGE_COLS]
    X = df[feature_cols].copy()
    y = df['Outcome'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clip_bounds = {}
    X_train_clean, X_test_clean = X_train.copy(), X_test.copy()
    num_feat_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    for col in num_feat_cols:
        Q1, Q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        clip_bounds[col] = (lower, upper)
        X_train_clean[col] = X_train[col].clip(lower, upper)
        X_test_clean[col] = X_test[col].clip(lower, upper)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  eval_metric='logloss', random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    }

    results, trained_models = {}, {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'y_pred': y_pred, 'y_prob': y_prob,
        }
        trained_models[name] = model

    best_model_name = max(results, key=lambda k: results[k]['ROC-AUC'])

    return {
        'feature_cols': feature_cols, 'scaler': scaler, 'clip_bounds': clip_bounds,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'X_test_scaled': X_test_scaled, 'results': results, 'trained_models': trained_models,
        'best_model_name': best_model_name,
    }


def scale_new_patient(patient_dict, artifacts):
    row = pd.DataFrame([patient_dict])[artifacts['feature_cols']]
    for col, (lower, upper) in artifacts['clip_bounds'].items():
        if col in row.columns:
            row[col] = row[col].clip(lower, upper)
    return artifacts['scaler'].transform(row)


# ------------------------------------------------------------------
# SIDEBAR — DATA SOURCE
# ------------------------------------------------------------------
st.sidebar.title("🩺 Diabetes Risk Explorer")
st.sidebar.markdown("**Pakistani Diabetes Dataset**")
uploaded = st.sidebar.file_uploader("Upload Pakistani_Diabetes_Dataset.csv", type=["csv"])

if uploaded is None:
    st.sidebar.info("Upload the dataset CSV to get started. The app expects the same columns/format used in the original notebook.")
    st.title("🩺 Pakistani Diabetes Risk Analysis")
    st.markdown(
        "This interactive app rebuilds your Data Visualization project — "
        "EDA, 5-model comparison, evaluation, and a live risk predictor.\n\n"
        "👈 **Upload `Pakistani_Diabetes_Dataset.csv` in the sidebar to begin.**"
    )
    st.stop()

try:
    df = load_and_clean(uploaded)
except Exception as e:
    st.error(f"Could not load/clean the dataset: {e}")
    st.stop()

required = ['Age', 'Gender', 'Rgn', 'Weight', 'BMI', 'Waist', 'Systolic_BP', 'Diastolic_BP',
            'Family_History', 'A1c', 'Blood_Sugar', 'Vision_Problems', 'Exr',
            'Polydipsia', 'Polyuria', 'Nephropathy', 'HDL_Cholesterol', 'Outcome']
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"These expected columns are missing after cleaning: {missing}")
    st.stop()

st.sidebar.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
st.sidebar.markdown("---")
st.sidebar.metric("Diabetes Prevalence", f"{df['Outcome'].mean()*100:.1f}%")

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_overview, tab_eda, tab_models, tab_eval, tab_predict = st.tabs(
    ["📋 Overview", "📊 EDA", "🤖 Model Training", "📈 Evaluation", "🔮 Predict Risk"]
)

# ==================== TAB 1: OVERVIEW ====================
with tab_overview:
    st.header("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    st.subheader("Sample Data")
    st.dataframe(df.head(20), width='stretch')

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().round(2), width='stretch')

    st.subheader("Class Distribution")
    colA, colB = st.columns(2)
    counts = df['Outcome_Label'].value_counts()
    with colA:
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                      color_discrete_map=PAL, labels={'x': '', 'y': 'Count'},
                      title="Class Distribution (Count)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with colB:
        fig = px.pie(values=counts.values, names=counts.index, color=counts.index,
                      color_discrete_map=PAL, title="Class Distribution (%)")
        st.plotly_chart(fig, width='stretch')

# ==================== TAB 2: EDA ====================
with tab_eda:
    st.header("Exploratory Data Analysis")

    st.subheader("Feature Distribution Explorer")
    cont_cols = [c for c in ['Age', 'BMI', 'Blood_Sugar', 'A1c', 'Systolic_BP', 'Diastolic_BP',
                              'Weight', 'Waist', 'HDL_Cholesterol', 'Exr'] if c in df.columns]
    sel_feature = st.selectbox("Choose a feature to explore:", cont_cols)
    fig = px.histogram(df, x=sel_feature, color='Outcome_Label', barmode='overlay',
                        color_discrete_map=PAL, marginal="box", opacity=0.6,
                        title=f"{sel_feature} Distribution by Outcome")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Correlation Heatmap")
    heatmap_df = df.select_dtypes(include=np.number)
    corr = heatmap_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto",
                     title="Correlation Heatmap — All Numeric Features")
    fig.update_layout(height=650)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Top correlations with Outcome"):
        st.dataframe(corr['Outcome'].drop('Outcome').sort_values(ascending=False).round(3))

    st.subheader("Gender Analysis")
    colA, colB, colC = st.columns(3)
    gender_labels = df['Gender'].map({0: 'Female', 1: 'Male'})
    with colA:
        fig = px.pie(values=gender_labels.value_counts().values, names=gender_labels.value_counts().index,
                      title="Gender Distribution", color_discrete_sequence=['#3498db', '#e91e63'])
        st.plotly_chart(fig, width='stretch')
    with colB:
        tmp = df.copy(); tmp['Gender_Label'] = gender_labels
        fig = px.histogram(tmp, x='Gender_Label', color='Outcome_Label', barmode='group',
                            color_discrete_map=PAL, title="Diabetes Count by Gender")
        st.plotly_chart(fig, width='stretch')
    with colC:
        rate = df.groupby('Gender')['Outcome'].mean() * 100
        rate.index = ['Female', 'Male']
        fig = px.bar(x=rate.index, y=rate.values, title="Diabetes Rate by Gender (%)",
                      labels={'x': '', 'y': 'Rate (%)'}, color=rate.index,
                      color_discrete_sequence=['#e91e63', '#3498db'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Age & BMI Analysis")
    colA, colB = st.columns(2)
    with colA:
        age_rate = df.groupby('Age_Group', observed=True)['Outcome'].mean() * 100
        fig = px.bar(x=age_rate.index.astype(str), y=age_rate.values,
                      title="Diabetes Rate by Age Group", labels={'x': 'Age Group', 'y': 'Rate (%)'},
                      color=age_rate.values, color_continuous_scale='YlOrRd')
        st.plotly_chart(fig, width='stretch')
    with colB:
        fig = px.scatter(df, x='Age', y='BMI', color='Outcome_Label', color_discrete_map=PAL,
                          opacity=0.6, title="Age vs BMI by Diabetes Outcome")
        st.plotly_chart(fig, width='stretch')

    if 'Rgn' in df.columns:
        st.subheader("Urban vs Rural Analysis")
        colA, colB = st.columns(2)
        rgn_rate = df.groupby('Rgn')['Outcome'].mean() * 100
        rgn_rate.index = ['Urban' if r == 0 else 'Rural' for r in rgn_rate.index]
        with colA:
            tmp = df.copy(); tmp['Rgn_Label'] = tmp['Rgn'].map({0: 'Urban', 1: 'Rural'})
            fig = px.histogram(tmp, x='Rgn_Label', color='Outcome_Label', barmode='group',
                                color_discrete_map=PAL, title="Diabetes Count by Region")
            st.plotly_chart(fig, width='stretch')
        with colB:
            fig = px.bar(x=rgn_rate.index, y=rgn_rate.values, title="Diabetes Rate by Region (%)",
                          color=rgn_rate.index, color_discrete_sequence=['#9b59b6', '#f39c12'],
                          labels={'x': '', 'y': 'Rate (%)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

    st.subheader("Risk Factor Analysis (Binary Features)")
    binary_cols = [c for c in ['Family_History', 'Vision_Problems', 'Polydipsia', 'Polyuria', 'Nephropathy']
                   if c in df.columns]
    sel_binary = st.multiselect("Choose risk factors:", binary_cols, default=binary_cols)
    if sel_binary:
        cols = st.columns(len(sel_binary))
        for col, c in zip(sel_binary, cols):
            rate = df.groupby(col)['Outcome'].mean() * 100
            fig = px.bar(x=[str(v) for v in rate.index], y=rate.values,
                          title=col.replace('_', ' '), labels={'x': 'No / Yes', 'y': 'Rate (%)'},
                          color_discrete_sequence=[C1])
            c.plotly_chart(fig, width='stretch')

    st.subheader("BMI Category Analysis")
    colA, colB = st.columns(2)
    with colA:
        bmi_counts = df['BMI_Category'].value_counts().sort_index()
        fig = px.bar(x=bmi_counts.index.astype(str), y=bmi_counts.values,
                      title="BMI Category Distribution", labels={'x': '', 'y': 'Count'})
        st.plotly_chart(fig, width='stretch')
    with colB:
        bmi_rate = df.groupby('BMI_Category', observed=True)['Outcome'].mean() * 100
        fig = px.bar(x=bmi_rate.index.astype(str), y=bmi_rate.values,
                      title="Diabetes Rate by BMI Category", labels={'x': '', 'y': 'Rate (%)'},
                      color=bmi_rate.values, color_continuous_scale='YlOrRd')
        st.plotly_chart(fig, width='stretch')

# ==================== TAB 3: MODEL TRAINING ====================
with tab_models:
    st.header("Model Training & Comparison")
    st.markdown(
        "Trains **Logistic Regression, Decision Tree, Random Forest, XGBoost, and SVM** "
        "on an 80/20 split, exactly like the notebook (outlier capping fit on train only, "
        "then scaled, with `A1c`, `Polydipsia`, `Polyuria`, and `Duration` excluded to avoid leakage)."
    )

    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training 5 models + running 5-fold cross-validation..."):
            st.session_state["artifacts"] = train_all_models(df)
        st.success("Training complete!")

    if "artifacts" in st.session_state:
        artifacts = st.session_state["artifacts"]
        results = artifacts['results']

        metrics_df = pd.DataFrame({
            name: {
                'Accuracy': round(r['Accuracy'], 4), 'Precision': round(r['Precision'], 4),
                'Recall': round(r['Recall'], 4), 'F1 Score': round(r['F1 Score'], 4),
                'ROC-AUC': round(r['ROC-AUC'], 4), 'CV Score': f"{r['CV Mean']:.3f} ± {r['CV Std']:.3f}",
            } for name, r in results.items()
        }).T.sort_values('ROC-AUC', ascending=False)

        st.subheader("Performance Comparison Table")
        st.dataframe(metrics_df, width='stretch')
        st.success(f"🏆 Best model (by ROC-AUC): **{artifacts['best_model_name']}**")

        st.subheader("Metrics Comparison Chart")
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        plot_rows = []
        for name, r in results.items():
            for m in metric_names:
                plot_rows.append({'Model': name, 'Metric': m, 'Score': r[m]})
        plot_df = pd.DataFrame(plot_rows)
        fig = px.bar(plot_df, x='Metric', y='Score', color='Model', barmode='group',
                      title="Model Performance Comparison")
        fig.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig, width='stretch')

        st.subheader("5-Fold Cross Validation Scores")
        cv_names = list(results.keys())
        cv_means = [results[n]['CV Mean'] for n in cv_names]
        cv_stds = [results[n]['CV Std'] for n in cv_names]
        fig = go.Figure(go.Bar(x=cv_names, y=cv_means, error_y=dict(type='data', array=cv_stds)))
        fig.update_layout(title="CV Accuracy (Mean ± Std)", yaxis_title="Accuracy")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Click **Train All Models** to run the pipeline.")

# ==================== TAB 4: EVALUATION ====================
with tab_eval:
    st.header("Model Evaluation")
    if "artifacts" not in st.session_state:
        st.warning("Train the models first in the **Model Training** tab.")
    else:
        artifacts = st.session_state["artifacts"]
        results = artifacts['results']
        model_names = list(results.keys())
        sel_model = st.selectbox("Select a model to evaluate:", model_names,
                                  index=model_names.index(artifacts['best_model_name']))
        r = results[sel_model]
        y_test = artifacts['y_test']

        colA, colB = st.columns(2)
        with colA:
            st.subheader(f"Confusion Matrix — {sel_model}")
            cm = confusion_matrix(y_test, r['y_pred'])
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                             labels=dict(x="Predicted", y="Actual"),
                             x=["Non-Diabetic", "Diabetic"], y=["Non-Diabetic", "Diabetic"])
            st.plotly_chart(fig, width='stretch')

        with colB:
            st.subheader("Classification Report")
            report = classification_report(y_test, r['y_pred'],
                                             target_names=["Non-Diabetic", "Diabetic"], output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3), width='stretch')

        st.subheader("ROC Curves — All Models")
        fig = go.Figure()
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={res['ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='gray'), name='Random'))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, width='stretch')

        st.subheader("Precision-Recall Curves — All Models")
        fig = go.Figure()
        for name, res in results.items():
            prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
            ap = average_precision_score(y_test, res['y_prob'])
            fig.add_trace(go.Scatter(x=rec, y=prec, name=f"{name} (AP={ap:.3f})"))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, width='stretch')

        st.subheader("Feature Importance")
        colA, colB = st.columns(2)
        with colA:
            if 'Random Forest' in artifacts['trained_models']:
                rf_model = artifacts['trained_models']['Random Forest']
                importances = pd.Series(rf_model.feature_importances_,
                                         index=artifacts['feature_cols']).sort_values()
                fig = px.bar(x=importances.values, y=importances.index, orientation='h',
                              title="Random Forest Feature Importance")
                st.plotly_chart(fig, width='stretch')
        with colB:
            if 'XGBoost' in artifacts['trained_models']:
                xgb_model = artifacts['trained_models']['XGBoost']
                xgb_importances = pd.Series(xgb_model.feature_importances_,
                                             index=artifacts['feature_cols']).sort_values()
                fig = px.bar(x=xgb_importances.values, y=xgb_importances.index, orientation='h',
                              title="XGBoost Feature Importance")
                st.plotly_chart(fig, width='stretch')

        st.subheader("SHAP Explainability (XGBoost)")
        if st.checkbox("Compute SHAP summary plot (may take a moment)"):
            with st.spinner("Computing SHAP values..."):
                xgb_model = artifacts['trained_models']['XGBoost']
                X_test_df = pd.DataFrame(artifacts['X_test_scaled'], columns=artifacts['feature_cols'])
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(X_test_df)
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_df, show=False)
                st.pyplot(fig, width='stretch')

# ==================== TAB 5: PREDICT ====================
with tab_predict:
    st.header("🔮 Live Diabetes Risk Prediction")
    if "artifacts" not in st.session_state:
        st.warning("Train the models first in the **Model Training** tab.")
    else:
        artifacts = st.session_state["artifacts"]
        model_names = list(artifacts['trained_models'].keys())
        sel_model_name = st.selectbox("Model to use for prediction:", model_names,
                                       index=model_names.index(artifacts['best_model_name']))

        st.markdown("Enter patient details:")
        col1, col2, col3 = st.columns(3)
        patient = {}
        feature_cols = artifacts['feature_cols']
        train_ref = artifacts['X_train']

        for i, feat in enumerate(feature_cols):
            target_col = [col1, col2, col3][i % 3]
            label = FEATURE_LABELS.get(feat, feat)
            series = train_ref[feat]
            unique_vals = sorted(series.dropna().unique().tolist())

            with target_col:
                if set(unique_vals).issubset({0, 1}):
                    choice = st.radio(label, options=[0, 1], horizontal=True, key=f"pred_{feat}")
                    patient[feat] = choice
                else:
                    lo, hi, med = float(series.min()), float(series.max()), float(series.median())
                    patient[feat] = st.slider(label, min_value=lo, max_value=hi, value=med, key=f"pred_{feat}")

        if st.button("Predict Risk", type="primary"):
            X_new_scaled = scale_new_patient(patient, artifacts)
            model = artifacts['trained_models'][sel_model_name]
            prob = model.predict_proba(X_new_scaled)[0, 1]
            pred = model.predict(X_new_scaled)[0]

            st.markdown("---")
            colA, colB = st.columns([1, 2])
            with colA:
                st.metric("Predicted Risk", f"{prob*100:.1f}%")
                if pred == 1:
                    st.error(f"⚠️ {sel_model_name} predicts: **Diabetic**")
                else:
                    st.success(f"✅ {sel_model_name} predicts: **Non-Diabetic**")
            with colB:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': "Diabetes Risk (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#e74c3c" if prob >= 0.5 else "#2ecc71"},
                           'steps': [{'range': [0, 50], 'color': "#eafaf1"},
                                     {'range': [50, 100], 'color': "#fdecea"}],
                           'threshold': {'line': {'color': "black", 'width': 3}, 'value': 50}}
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig, width='stretch')

            st.caption(
                "⚠️ This is an educational/prototype tool trained on a single dataset. "
                "It is **not** a medical diagnostic device — external validation is required "
                "before any real-world or clinical use (as noted in the notebook's conclusions)."
            )
