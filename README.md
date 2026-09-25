# Diabetes Hospital Readmission Prediction

This project predicts whether a diabetic patient is at risk of being readmitted to the hospital within 30 days.

The project includes complete data preprocessing, exploratory data analysis, feature engineering, imbalance handling, model training, hyperparameter tuning, ensemble learning, final evaluation, and deployment using FastAPI and Streamlit.

---

## Project Objective

The main objective is to identify patients who are at higher risk of hospital readmission within 30 days.

Because the target class is imbalanced, the project focuses on metrics such as:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Accuracy alone is not considered sufficient for evaluating model performance.

---

## Dataset

The project uses the diabetic hospital readmission dataset.

Main target:

- `1` = Readmitted within 30 days
- `0` = Not readmitted within 30 days

Patient-level splitting is used to reduce data leakage between Training, Validation, and Test sets.

---

## Machine Learning Pipeline

The project includes:

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Missing Value Handling
5. Outlier Analysis
6. Categorical Encoding
7. Numerical Scaling
8. Patient-Level Train / Validation / Test Split
9. Baseline Models
10. Class Imbalance Handling
11. SMOTE Experiments
12. Feature Engineering
13. Feature Selection
14. XGBoost Hyperparameter Tuning
15. LightGBM Hyperparameter Tuning
16. Threshold Optimization
17. Probability Ensemble
18. Weighted Ensemble
19. Final Test Evaluation
20. Deployment

---

## Models Evaluated

The following approaches were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- Weighted XGBoost
- LightGBM
- Balanced LightGBM
- SMOTE + XGBoost
- Partial SMOTE
- XGBoost with Engineered Features
- Feature-Selected XGBoost
- Tuned XGBoost
- Tuned LightGBM
- XGBoost + LightGBM Ensemble
- Weighted Probability Ensemble

---

## Final Model

The final selected model is a weighted probability ensemble:

- `60%` Tuned LightGBM
- `40%` XGBoost with Engineered Features
- Final classification threshold = `0.13`

The ensemble configuration was selected using Validation data only.

The Test set was kept untouched until final evaluation.

---

## Final Test Results

The final weighted ensemble achieved:

| Metric | Score |
|---|---:|
| Precision | 0.1968 |
| Recall | 0.4986 |
| F1-score | 0.2822 |
| ROC-AUC | 0.6768 |
| PR-AUC | 0.2390 |
| Accuracy | 0.7161 |

### Final Confusion Matrix

- True Negatives: `13,075`
- False Positives: `4,511`
- False Negatives: `1,111`
- True Positives: `1,105`

The model identifies approximately half of the actual `<30-day readmission` cases.

---

## Final Model Bundle

The trained deployment bundle is saved as:

```text
final_readmission_ensemble.joblib