# Model selection by five-fold cross-validation in the BRFSS 2020 development sample.
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'code')
from harmon import feats, NUM
import lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
b=pd.read_pickle('out/serviks2020.pkl'); y=b.y.astype(int).values; w=(b._LLCPWT/b._LLCPWT.mean()).values
TOOL=['yas','cocuk','irk','egitim','gelir8','medeni','istihdam','ev_sahibi','kirsal','dil_ispanyolca','sigortasiz','doktor','maliyet_engeli','checkup','genel_saglik','sigara','grip_asisi','hiv_testi','dis_hekimi']
XF=feats(b,2020); X=XF[TOOL]
knots=[23,26,30,40,50,60]
def design(X):
    P=[pd.DataFrame({'yas':X.yas-21,**{f'yas_k{k}':np.clip(X.yas-k,0,None) for k in knots},'cocuk':X.cocuk.fillna(0)})]
    for v in TOOL:
        if v in('yas','cocuk'): continue
        lv=sorted(X[v].dropna().unique()); P.append(pd.DataFrame({f'{v}={int(c)}':(X[v]==c).astype(float) for c in lv}|{f'{v}=NA':X[v].isna().astype(float)}))
    return pd.concat(P,axis=1)
Z=design(X).values.astype('float32')
def lgbfit(Xd,tr,te):
    XL=Xd.copy()
    for f in XL.columns:
        if f not in NUM: XL[f]=XL[f].astype('category')
    par=dict(objective='multiclass',num_class=3,learning_rate=0.05,num_leaves=31,min_child_samples=80,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5,cat_smooth=30,verbose=-1,n_jobs=2,seed=42)
    return lgb.train(par,lgb.Dataset(XL.iloc[tr],y[tr],weight=w[tr]),180).predict(XL.iloc[te])
R=[]
for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=7).split(Z,y)):
    # Fit preprocessing inside each fold so the validation fold cannot influence the MLP.
    scaler=StandardScaler().fit(Z[tr]); Ztr=scaler.transform(Z[tr]); Zte=scaler.transform(Z[te])
    P={}
    P['Lojistik regresyon']=LogisticRegression(C=1.0,max_iter=1500).fit(Z[tr],y[tr],sample_weight=w[tr]).predict_proba(Z[te])
    P['Karar ağacı']=DecisionTreeClassifier(max_depth=8,min_samples_leaf=200,random_state=0).fit(Z[tr],y[tr],sample_weight=w[tr]).predict_proba(Z[te])
    P['Random forest']=RandomForestClassifier(n_estimators=300,min_samples_leaf=20,n_jobs=2,random_state=0).fit(Z[tr],y[tr],sample_weight=w[tr]).predict_proba(Z[te])
    P['XGBoost']=xgb.XGBClassifier(n_estimators=350,learning_rate=0.05,max_depth=5,subsample=0.8,colsample_bytree=0.7,reg_lambda=5,n_jobs=2,tree_method='hist',random_state=0).fit(Z[tr],y[tr],sample_weight=w[tr]).predict_proba(Z[te])
    P['Yapay sinir ağı (MLP)']=MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-2,early_stopping=True,n_iter_no_change=5,max_iter=60,random_state=0).fit(Ztr,y[tr]).predict_proba(Zte)
    P['LightGBM']=lgbfit(X,tr,te)
    P['LightGBM (tüm 37 değişken + eyalet)']=lgbfit(XF,tr,te)
    for m,p in P.items():
        r=dict(model=m,kat=fold,logloss=log_loss(y[te],p,sample_weight=w[te]))
        for k,l in enumerate(['guncel','geri','hic']): r['AUC_'+l]=roc_auc_score(y[te]==k,p[:,k],sample_weight=w[te])
        r['AUC_macro']=np.mean([r['AUC_guncel'],r['AUC_geri'],r['AUC_hic']]); R.append(r)
    print('kat',fold,{m:round(np.mean([roc_auc_score(y[te]==k,p[:,k],sample_weight=w[te]) for k in range(3)]),4) for m,p in P.items()},flush=True)
R=pd.DataFrame(R); R.to_csv('out/final_cv_kat.csv',index=False)
S=R.groupby('model').agg(['mean','std']).drop(columns='kat'); S.columns=['_'.join(c) for c in S.columns]; S=S.sort_values('AUC_macro_mean',ascending=False); S.round(4).to_csv('out/final_cv.csv'); print(S.round(4).to_string())
