"""Compare classifiers using the same preprocessing as the application."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
from churn_predictor import load_training_data, build_pipeline


def run(output='results'):
    output=Path(output); output.mkdir(exist_ok=True)
    X,y=load_training_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    base=build_pipeline()
    models={'Decision Tree':DecisionTreeClassifier(class_weight='balanced',random_state=42),
        'Random Forest':RandomForestClassifier(class_weight='balanced',random_state=42,n_jobs=4),
        'LightGBM':base.named_steps['model']}
    rows=[]
    import matplotlib.pyplot as plt
    roc_fig,roc_ax=plt.subplots(); pr_fig,pr_ax=plt.subplots()
    for name,model in models.items():
        pipe=clone(base).set_params(model=clone(model))
        cv=cross_validate(pipe,X_train,y_train,cv=StratifiedKFold(5,shuffle=True,random_state=42),
            scoring={'ap':'average_precision','roc_auc':'roc_auc','recall':'recall','f1':'f1'},n_jobs=1)
        pipe.fit(X_train,y_train); pred=pipe.predict(X_test); probability=pipe.predict_proba(X_test)[:,1]
        rows.append({'model':name, 'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred),
            'recall':recall_score(y_test,pred),'f1':f1_score(y_test,pred),'roc_auc':roc_auc_score(y_test,probability),
            'average_precision':average_precision_score(y_test,probability),
            **{k.replace('test_', 'cv_'):float(np.mean(v)) for k,v in cv.items() if k.startswith('test_')}})
        RocCurveDisplay.from_predictions(y_test,probability,name=name,ax=roc_ax)
        PrecisionRecallDisplay.from_predictions(y_test,probability,name=name,ax=pr_ax)
    roc_fig.tight_layout(); pr_fig.tight_layout()
    roc_fig.savefig(output/'roc_curves.png'); pr_fig.savefig(output/'precision_recall.png'); plt.close('all')
    summary={'rows':len(y),'train_rows':len(y_train),'test_rows':len(y_test),'test_churners':int(y_test.sum()),
        'protocol':'Stratified 80/20 holdout, 5-fold stratified training CV. Imputation and encoding fit inside each fold.'}
    from importlib.metadata import version
    summary['versions']={p:version(p) for p in ['pandas','numpy','scikit-learn','lightgbm']}
    summary['input_sha256']=__import__('hashlib').sha256((Path(__file__).parent/'Telco Customer Churn.csv').read_bytes()).hexdigest()
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    result=pd.DataFrame(rows); result.to_csv(output/'metrics.csv',index=False); print(result.to_string(index=False))
    return result


if __name__=='__main__': run()
