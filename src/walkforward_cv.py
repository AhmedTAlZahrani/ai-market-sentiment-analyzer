import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

P = Path('data/processed')
FEATS = P / 'features_dataset.csv'
OUT   = P / 'model_cv_results.csv'
PRED  = P / 'predictions_nextday.csv'

def wf_splits(n, n_folds=4, min_train=40):
    step = max(1, (n - min_train)//n_folds)
    for k in range(n_folds):
        s = min_train + k*step
        e = min(n, s+step)
        yield slice(0,s), slice(s,e)

def train_eval(df, feat_cols):
    models = {'Ridge':Ridge(alpha=1.0),
              'XGB':XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08)}
    results=[]
    preds=[]
    for name, model in models.items():
        y_true, y_pred = [], []
        for tr, te in wf_splits(len(df)):
            Xtr, ytr = df.iloc[tr][feat_cols], df.iloc[tr]['TargetNext']
            Xte, yte = df.iloc[te][feat_cols], df.iloc[te]['TargetNext']
            if len(Xtr)<20 or len(Xte)<5: continue
            model.fit(Xtr, ytr)
            p = model.predict(Xte)
            y_true.extend(yte); y_pred.extend(p)
        if y_true:
            results.append({'Model':name,
                            'R2':round(r2_score(y_true,y_pred),3),
                            'MAE':round(mean_absolute_error(y_true,y_pred),3)})
    return results

def main():
    if not FEATS.exists():
        print("âš ï¸ Run features_signal_engine.py first."); return
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    feat_cols = [c for c in df.columns if c not in ['Date','Ticker','Close','Return','TargetNext']]
    rows=[]
    preds=[]
    for t in df['Ticker'].unique():
        sub = df[df['Ticker']==t].dropna(subset=feat_cols+['TargetNext'])
        res = train_eval(sub, feat_cols)
        for r in res:
            rows.append({'Ticker':t, **r})
    if rows:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f'âœ… WF-CV results saved to {OUT}')
    else:
        print('âš ï¸ No results.')
if __name__=='__main__':
    main()
