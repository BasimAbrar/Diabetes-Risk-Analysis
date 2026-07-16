# 🇵🇰 Pakistan Diabetes Risk Analysis using Machine Learning

A comprehensive **Machine Learning and Data Visualization** project that analyzes diabetes risk factors in the **Pakistani population**. This project combines Exploratory Data Analysis (EDA), feature engineering, data preprocessing, and predictive modeling to understand the factors associated with diabetes while following best practices to eliminate data leakage.

> **Note:** This project is built specifically using a **Pakistani diabetes dataset**. The analysis, insights, and model performance are intended for the Pakistani population and should not be generalized to other populations without further validation.

---

## 📌 Project Overview

The primary objective of this project is to:

- Explore diabetes-related health indicators within the Pakistani population.
- Perform comprehensive Exploratory Data Analysis (EDA).
- Visualize relationships between patient attributes and diabetes.
- Build and compare multiple Machine Learning models.
- Identify the most influential predictors of diabetes.
- Develop a reliable prediction pipeline by preventing data leakage.

---

## 📊 Features

- Data Cleaning & Preprocessing
- Missing Value Handling
- Outlier Detection & Treatment
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Correlation Analysis
- Feature Engineering
- Machine Learning Model Training
- Cross Validation
- Hyperparameter Tuning
- Model Comparison
- Feature Importance Analysis
- ROC Curves & Confusion Matrices
- Model Evaluation using multiple metrics

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## 📁 Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Outlier Handling
6. Train/Test Split
7. Machine Learning Model Training
8. Model Evaluation
9. Feature Importance Analysis
10. Final Model Selection

---

# 🔒 Data Leakage Fixes Applied

One of the biggest challenges during this project was identifying and removing features that caused **data leakage**. Some variables contained information that would only be available **after diagnosis** or were themselves diagnostic criteria, leading to unrealistically high model performance.

The following fixes were applied:

| Fix | Column Removed | Reason |
|------|----------------|--------|
| FIX 1 | Duration | Post-diagnosis variable |
| FIX 2 | Blood_Sugar | Direct diagnostic criterion (≥126 mg/dL) |
| FIX 3 | A1c (HbA1c) | Direct diagnostic criterion (≥6.5%) |
| FIX 4 | Polydipsia | Diabetes symptom with ~92.2% diabetes rate |
| FIX 5 | Polyuria | Diabetes symptom with ~92.0% diabetes rate |
| FIX 6 | Pipeline Order | Outlier capping performed **after** train/test split using only training data |

---

## ⚠️ Data Leakage Note

The following columns **exist in the original dataset** but were **excluded from model training**:

- Blood_Sugar
- A1c (HbA1c)
- Polydipsia
- Polyuria
- Duration

These variables remain in the **EDA section** to provide useful clinical insights but are **never used as input features** for the Machine Learning models.

This ensures that the reported performance reflects how the model would behave in a real-world prediction scenario.

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Cross Validation
- Confusion Matrix
- ROC Curve

---

## 🎯 Project Goal

The goal of this project is not only to build an accurate Machine Learning model but also to create a **realistic, unbiased, and clinically meaningful prediction pipeline** by following proper Machine Learning practices and avoiding data leakage.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

The trained models should **not be used for clinical diagnosis or medical decision-making** without proper medical validation and approval.

---
