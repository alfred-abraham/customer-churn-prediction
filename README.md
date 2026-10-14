# Customer Churn Prediction

## Overview

A telecommunications churn classification project with a Streamlit application for predicting churn from customer account and service details.

## Objective

Identify customers at risk of leaving so retention efforts can prioritize likely churners. Compare Decision Tree, Random Forest, and LightGBM models using multiple classification metrics.

## Data

The IBM Telco Customer Churn dataset, distributed through Kaggle, contains **7,043 customers** and demographic, service, account, billing, and churn fields. The repository includes [the CSV](Telco%20Customer%20Churn.csv).

## Approach

1. Inspect data quality, feature types, and class imbalance.
2. Remove customer IDs, convert `TotalCharges` to numeric, fill missing values with the median, and encode categorical variables.
3. Split the data into training and test sets.
4. Compare three classifiers using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
5. Examine feature importance and connect the findings to retention opportunities.

See [the analysis notebook](Customer_Churn.ipynb) for the complete workflow.

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 74.1% | 0.51 | 0.49 | 0.50 | 0.662 |
| Random Forest | 79.9% | 0.67 | 0.47 | 0.55 | 0.840 |
| LightGBM | 77.0% | 0.54 | 0.75 | 0.63 | 0.853 |

LightGBM was selected for its higher recall and ROC-AUC, while Random Forest had the highest accuracy. Monthly charges, total charges, and tenure were prominent predictors; these findings support investigating targeted retention for newer customers and those with high monthly charges.

## Tools

Python, pandas, NumPy, scikit-learn, LightGBM, Matplotlib, Jupyter Notebook, and Streamlit.

## How to Run

Clone the repository and enter its directory:

```bash
git clone https://github.com/alfred-abraham/customer-churn-prediction.git
cd customer-churn-prediction
```

Create and activate a virtual environment:

- **Windows PowerShell:** `python -m venv .venv`, then `.\.venv\Scripts\Activate.ps1`.
- **macOS/Linux:** `python3 -m venv .venv`, then `source .venv/bin/activate`.

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open [localhost:8501](http://localhost:8501). Enter customer details to receive a classification and estimated probability. The first prediction trains the model on the included dataset and caches it for the app process; training and inference share preprocessing.

For the analysis, open [`Customer_Churn.ipynb`](Customer_Churn.ipynb) in Jupyter or Google Colab and change its `/content/Telco Customer Churn.csv` path to your CSV location before running cells.

To deploy on Streamlit Community Cloud, select this repository and set `app.py` as the entry point. No secrets or external services are required.

## Limitations and Next Steps

- Reported performance describes the project test set; it does not establish effectiveness for another customer population.
- Feature importance describes predictive associations rather than causal drivers.
- Future work includes hyperparameter tuning, precision-recall AUC, and SHAP explanations.

## License

See [LICENSE](LICENSE) for repository licensing. Refer to the dataset source for its usage terms.
