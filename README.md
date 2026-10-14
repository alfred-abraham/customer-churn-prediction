# Customer churn prediction

Compare Decision Tree, Random Forest and LightGBM classifiers on the 7,043-row [IBM Telco churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The Streamlit app uses the same preprocessing and LightGBM configuration as the evaluation.

## Run
Python 3.12:
```bash
python -m pip install -r requirements-dev.txt
python evaluate_models.py
python -m pytest tests
python -m streamlit run app.py
```
Open `Customer_Churn.ipynb` to inspect the saved evaluation; set RECOMPUTE=True to rerun. On Windows, create and activate a virtual environment with `python -m venv .venv` and `.\.venv\Scripts\Activate.ps1`.

## Method
Drop customerID, map the churn target to 0/1, convert TotalCharges to numeric, and median-impute numerical features inside a pipeline. Categorical features are imputed and one-hot encoded. Fit transformations independently within each of five stratified CV training folds. The stratified 80/20 holdout is separate from training CV.
Class weights emphasize churn recall. LightGBM remains the preferred app model based on training CV recall and average precision; holdout metrics are reported separately.

| model | accuracy | precision | recall | f1 | roc_auc | average_precision | cv_ap | cv_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | 0.7296 | 0.4907 | 0.4920 | 0.4913 | 0.6542 | 0.3775 | 0.3826 | 0.4903 |
| Random Forest | 0.7736 | 0.5653 | 0.6364 | 0.5987 | 0.8255 | 0.6109 | 0.6171 | 0.6475 |
| LightGBM | 0.7573 | 0.5301 | 0.7540 | 0.6225 | 0.8324 | 0.6413 | 0.6502 | 0.7338 |

The `cv_ap` and `cv_recall` columns are mean training-CV validation scores; the other columns describe the final holdout. Complete metrics and protocol metadata are in [results](results/).
ROC and precision–recall curves are included in the notebook. The old metrics and preprocessing mismatch are superseded.

## Application behavior
The first prediction trains on all labeled repository rows and caches the pipeline. Evaluation uses a separate training partition; fitting the final demonstration model on all rows is not itself an evaluation. No secrets are needed. For Streamlit Community Cloud, use `app.py` as the entry point.

## Interpretation
High-risk customers can be candidates for a retention review, but associations do not establish that changing charges or contract terms causes retention. The class-weighted probability output is uncalibrated; treat it as a model score. Select an operational threshold with validation data and campaign costs, then evaluate it once on independent data. The dataset is a single historical snapshot, and no time-based or external validation is claimed.
