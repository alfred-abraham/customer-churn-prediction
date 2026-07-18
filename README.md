# Customer Churn Prediction using Machine Learning

## Project Overview

Customer retention is often more cost-effective than acquiring new customers. This project uses machine learning to predict customer churn for a telecommunications company using the Telco Customer Churn dataset.

Three classification models were developed and compared:

- Decision Tree
- Random Forest
- LightGBM

The models were evaluated using multiple classification metrics to determine which model best identifies customers likely to churn. Feature importance analysis was also performed to identify the business factors that contribute most to customer attrition.

---

## Business Problem

Telecommunication companies lose revenue when customers discontinue their services. The goal of this project is to develop a predictive model capable of identifying customers at high risk of churn so retention strategies can be implemented before customers leave.

---

## Dataset

**Source:** IBM Telco Customer Churn Dataset (Kaggle)
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains information on **7,043 customers**, including:

- Customer demographics
- Account information
- Services subscribed
- Billing information
- Churn status

Target variable:

- **Churn**
    - Yes
    - No

---

## Project Workflow

### 1. Data Exploration

- Inspected dataset structure and data types
- Checked for missing values
- Checked for duplicate records
- Examined class imbalance
- Identified categorical and numerical variables

---

### 2. Data Preprocessing

Performed the following preprocessing steps:

- Removed the customer ID column
- Converted `TotalCharges` to numeric
- Filled missing values using the median
- Converted the target variable into binary values
- Applied One-Hot Encoding to categorical variables
- Split the data into training and testing sets

---

### 3. Machine Learning Models

Three supervised learning models were trained and evaluated:

- Decision Tree
- Random Forest
- LightGBM

---

### 4. Evaluation Metrics

Models were compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|---------:|----------:|--------:|---------:|---------:|
| Decision Tree | 74.1% | 0.51 | 0.49 | 0.50 | 0.662 |
| Random Forest | **79.9%** | **0.67** | 0.47 | 0.55 | 0.840 |
| LightGBM | 77.0% | 0.54 | **0.75** | **0.63** | **0.853** |

### Best Model

Although Random Forest achieved the highest overall accuracy, LightGBM produced substantially higher recall and the strongest ROC-AUC.

Because customer churn prediction prioritizes identifying customers likely to leave, **LightGBM was selected as the preferred model**.

---

## Feature Importance

Feature importance analysis identified the following variables as the strongest predictors of churn:

- Monthly Charges
- Total Charges
- Customer Tenure

Because categorical variables were one-hot encoded during preprocessing, feature importances for related dummy variables were aggregated to better represent the contribution of the original business variables.

---

## Business Insights

The analysis suggests that customer churn is primarily influenced by:

- Higher monthly charges
- Lower customer tenure
- Overall customer spending

These findings indicate that newer customers and customers with relatively high monthly charges may benefit from proactive retention strategies.

Potential business actions include:

- Offering incentives to newer customers
- Monitoring customers with high monthly charges
- Using the model to identify high-risk customers for targeted retention campaigns

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- Matplotlib
- Jupyter Notebook

---

## Key Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- One-Hot Encoding
- Supervised Machine Learning
- Model Evaluation
- ROC-AUC Analysis
- Feature Importance Analysis
- Business Insight Generation

---

## Conclusion

Three machine learning models were developed to predict customer churn.

Random Forest achieved the highest accuracy, while LightGBM demonstrated superior recall and ROC-AUC, making it the most suitable model for customer retention applications.

The project also identified monthly charges, total charges, and customer tenure as the strongest predictors of churn, providing actionable business insights that could help reduce customer attrition.

This project demonstrates an end-to-end machine learning workflow, from data preprocessing through model evaluation and interpretation, while translating technical findings into business recommendations.

---
## Future Improvements

- Perform hyperparameter tuning using GridSearchCV or Optuna.
- Evaluate Precision-Recall AUC for the imbalanced classification problem.
- Deploy the LightGBM model as an interactive Streamlit application.
- Investigate SHAP values for more detailed model interpretability.
