# Customer Churn Prediction

This project uses machine learning to predict whether a customer will churn based on demographic, billing, and service-related features. Predicting churn helps businesses identify customers at risk of leaving and implement strategies to improve customer retention.

The project compares multiple classification models to determine which approach performs best for predicting churn.

---
## Skills Demonstrated

- Data preprocessing
- Feature engineering (one-hot encoding)
- Feature scaling
- Model training and comparison
- Evaluation using accuracy and F1-score
- Feature importance analysis

---
## Dataset

The dataset contains **1000 customer records** with information about:
- CustomerID
- Age
- Gender
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Tech Support
- Internet Service
- Churn (target variable)

The goal of this project is to build a model that can accurately predict whether a customer will churn.

---

## Project Workflow

The notebook follows a typical machine learning pipeline:

1. Data Loading
2. Data Preprocessing
3. Feature Engineering
4. Train/Test Split
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Feature Importance Analysis

---

## Feature Engineering

Categorical variables were transformed using **one-hot encoding** to convert them into numerical features suitable for machine learning models.

Examples of engineered features include:

- `ContractType_One-Year`
- `ContractType_Two-Year`
- `TechSupport_Yes`
- `Gender_Male`

Feature scaling was applied using **StandardScaler** to improve performance for models such as Logistic Regression and Support Vector Machines.

---

## Models Implemented

Three classification models were trained and compared.

### Logistic Regression
A linear model commonly used as a baseline for classification problems.

### Support Vector Machine (SVM)
A margin-based classifier capable of learning nonlinear decision boundaries.

### Random Forest
An ensemble learning method that combines multiple decision trees to capture complex relationships between features.

---

## Model Evaluation

Models were evaluated using the following metrics:

- Accuracy
- F1-score

### Results

| Model | Accuracy | Churn F1 Score | Non-Churn F1 Score |
|------|------|------|------|
| Logistic Regression | 0.94 | 0.96 | 0.79 |
| Support Vector Machine | 0.97 | 0.98 | 0.88 |
| Random Forest | **0.995** | **1.00** | **0.98** |

The **Random Forest classifier** achieved the best performance, indicating that ensemble tree-based models are highly effective for this dataset.

---

## Feature Importance

Feature importance analysis from the Random Forest model revealed the most influential predictors of churn.

Top contributing features include:

- Tech Support availability
- Customer tenure
- Monthly charges

The results suggest that **service experience and contract structure play a greater role in churn than demographic characteristics such as age or gender**.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook / Google Colab

---

## Future Improvements

Potential improvements for this project include:

- Hyperparameter tuning using GridSearchCV
- Cross-validation for more robust evaluation
- Handling class imbalance with resampling techniques
- ROC curve and AUC comparison between models
- Model deployment as a simple prediction API


