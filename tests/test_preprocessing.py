import numpy as np
from churn_predictor import load_training_data,build_pipeline

def test_imputation_fits_training_only_and_handles_unseen_categories():
    X,y=load_training_data()
    train=X.iloc[:20].copy()
    train['TotalCharges']=np.arange(20,dtype=float)
    train.loc[train.index[0],'TotalCharges']=np.nan
    pipeline=build_pipeline()
    pipeline.fit(train,y.iloc[:20])
    assert pipeline.named_steps['preprocessing'].named_transformers_['numeric'].statistics_[-1]==10.0
    test=X.iloc[[21]].copy(); test['TotalCharges']=np.nan; test['gender']='unseen'
    probability=pipeline.predict_proba(test)
    assert np.isfinite(probability).all()
    assert np.isclose(probability.sum(),1)
