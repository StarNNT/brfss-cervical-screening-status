# Tuned boosting models and ensembles on the held-out 2020 set and the 2024 set, confusion matrices under three assignment rules, SHAP and coefficient-based importance, odds ratios. Run after 10.
# Hoca istekleri: (1) genişletilmiş algoritma karşılaştırması (hiperparametre araması, CatBoost, topluluk) — geliştirme kümesinde 5 katlı CV,
# (2) ayrılmış test (2020) ve dış (2024) doğrulama, (3) confusion matrix'ler, (4) SHAP + göreli önem. Çıktı: out/hoca.json, out/hoca_*.npy
import pandas as pd, numpy as np, sys, json, warnings, time; warnings.filterwarnings('ignore'); sys.path.insert(0,'code')
from harmon import feats, NUM
import lightgbm as lgb, xgboost as xgb, shap
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
b=pd.read_pickle('out/serviks2020.pkl').reset_index(drop=True); y=b.y.astype(int).values; w=(b._LLCPWT/b._LLCPWT.mean()).values
a=pd.read_pickle('out/analitik.pkl').reset_index(drop=True); ya=a.y.map({'current':0,'overdue':1,'never':2}).values; wa=(a._LLCPWT/a._LLCPWT.mean()).values
TOOL=['yas','cocuk','irk','egitim','gelir8','medeni','istihdam','ev_sahibi','kirsal','dil_ispanyolca','sigortasiz','doktor','maliyet_engeli','checkup','genel_saglik','sigara','grip_asisi','hiv_testi','dis_hekimi']
XF=feats(b,2020); XA=feats(a,2024); X=XF[TOOL]; X24=XA[TOOL]
src=open('code/03_final_model_validation_export.py').read(); exec(src[src.index("V={'irk'"):src.index("Z=design(XF)")])
Z=design(XF); Z24=design(XA); cols=list(Z.columns); Z=Z.values.astype('float32'); Z24=Z24[cols].values.astype('float32')
mu,sd=Z.mean(0),Z.std(0)+1e-9; Zs=(Z-mu)/sd; Z24s=(Z24-mu)/sd
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
CAT=[c for c in TOOL if c not in NUM]
def tocat(Xd):
    XL=Xd.copy()
    for f in XL.columns:
        if f not in NUM: XL[f]=XL[f].astype('category')
    return XL
XL=tocat(X); XL24=tocat(X24)
for c in CAT: XL24[c]=pd.Categorical(XL24[c],categories=XL[c].cat.categories)
def cb_frame(Xd):
    D=Xd.copy()
    for c in CAT: D[c]=D[c].fillna(-1).astype(int).astype(str)
    return D
XC=cb_frame(X); XC24=cb_frame(X24)
def metrics(t,p,ww):
    r={'logloss':float(log_loss(t,p,sample_weight=ww,labels=[0,1,2]))}
    for k,l in enumerate(['current','overdue','never']): r['auc_'+l]=float(roc_auc_score(t==k,p[:,k],sample_weight=ww))
    r['auc_macro']=float(np.mean([r['auc_current'],r['auc_overdue'],r['auc_never']])); return r
LOG=open('out/hoca.log','a');
def log(*x): print(*x,flush=True); LOG.write(' '.join(map(str,x))+'\n'); LOG.flush()
# ---------- modeller ----------
def fit_lr(itr): return LogisticRegression(C=1.0,max_iter=3000).fit(Z[itr],y[itr],sample_weight=w[itr])
def fit_lgb(itr,par,nround):
    p=dict(objective='multiclass',num_class=3,verbose=-1,n_jobs=2,seed=42,subsample_freq=1)|par
    return lgb.train(p,lgb.Dataset(XL.iloc[itr],y[itr],weight=w[itr]),nround)
def fit_xgb(itr,par): return xgb.XGBClassifier(n_jobs=2,tree_method='hist',random_state=0,**par).fit(Z[itr],y[itr],sample_weight=w[itr])
def fit_cb(itr,par): return CatBoostClassifier(loss_function='MultiClass',verbose=0,thread_count=2,random_seed=0,cat_features=CAT,**par).fit(XC.iloc[itr],y[itr],sample_weight=w[itr])
LGB0=dict(learning_rate=0.05,num_leaves=31,min_child_samples=80,subsample=0.8,colsample_bytree=0.7,reg_lambda=5,cat_smooth=30); XGB0=dict(n_estimators=350,learning_rate=0.05,max_depth=5,subsample=0.8,colsample_bytree=0.7,reg_lambda=5)
SR=json.load(open('out/search_results.json')); CVbest={k:{'auc_macro':v['auc_macro'][0],'logloss':v['logloss'][0]} for k,v in SR['cv'].items()}
BP=json.load(open('out/search_best.json')); best={k:(None,BP[k]) for k in ['lgb','xgb','cb']}
json.dump(dict(cv_best=CVbest,best=BP),open('out/hoca_cv.json','w'),indent=1)
# ---------- held-out test (models fitted to the 80% development subset) + 2024 (models refitted to the FULL 2020 sample) ----------
# Hyperparameters are those selected by CV in the development subset (out/search_best.json); the held-out predictions come from
# models fitted to the development subset only, whereas the 2024 (and deployment) predictions come from models refitted to all of
# 2020 with the same hyperparameters, as described in the paper. The refitted LR is identical to the one exported by script 03.
ALL=np.arange(len(y)); bl=best['lgb'][1]; blp={k:v for k,v in bl.items() if k!='nround'}
lr=fit_lr(tr); lrF=fit_lr(ALL); P={'Multinomial logistic regression':(lr.predict_proba(Z[te]),lrF.predict_proba(Z24))}
g=fit_lgb(tr,blp,bl['nround']); gF=fit_lgb(ALL,blp,bl['nround']); P['LightGBM (tuned)']=(g.predict(XL.iloc[te]),gF.predict(XL24))
xg=fit_xgb(tr,best['xgb'][1]); xgF=fit_xgb(ALL,best['xgb'][1]); P['XGBoost (tuned)']=(xg.predict_proba(Z[te]),xgF.predict_proba(Z24))
cbm=fit_cb(tr,best['cb'][1]); cbF=fit_cb(ALL,best['cb'][1]); P['CatBoost (tuned)']=(cbm.predict_proba(XC.iloc[te]),cbF.predict_proba(XC24))
P['Ensemble (mean of LR, LightGBM, XGBoost, CatBoost)']=(np.mean([P[k][0] for k in list(P)],0),np.mean([P[k][1] for k in list(P)],0))
P['Ensemble (mean of LightGBM, XGBoost, CatBoost)']=(np.mean([P[k][0] for k in ['LightGBM (tuned)','XGBoost (tuned)','CatBoost (tuned)']],0),np.mean([P[k][1] for k in ['LightGBM (tuned)','XGBoost (tuned)','CatBoost (tuned)']],0))
rsb=np.random.RandomState(5); KEYS=list(P); HO={k:{'internal':metrics(y[te],P[k][0],w[te]),'external_2024':metrics(ya,P[k][1],wa)} for k in KEYS}
for k in KEYS: log(k,json.dumps(HO[k]['internal']),json.dumps(HO[k]['external_2024']))
json.dump(HO,open('out/hoca_holdout.json','w'),indent=1)
# Paired model differences with design-based (within-stratum) bootstrap intervals are computed in script 13 from out/hoca_pred.npz.
json.dump(HO,open('out/hoca_holdout.json','w'),indent=1)
np.savez('out/hoca_pred.npz',**{f'{i}_{s}':P[k][j] for i,k in enumerate(P) for j,s in enumerate(['te','24'])})
# ---------- confusion matrix'ler ----------
def assign_thr(p,tn=0.10,to=0.15): return np.where(p[:,2]>=tn,2,np.where(p[:,1]>=to,1,0))
def cm(t,pred,ww):
    M=np.zeros((3,3))
    for i in range(3):
        for j in range(3): M[i,j]=ww[(t==i)&(pred==j)].sum()
    M=M/ww.sum()*100; rec=[M[i,i]/M[i].sum()*100 for i in range(3)]; prec=[M[i,i]/M[:,i].sum()*100 if M[:,i].sum()>0 else 0 for i in range(3)]
    f1=[2*rec[i]*prec[i]/(rec[i]+prec[i]) if rec[i]+prec[i]>0 else 0 for i in range(3)]
    return dict(matrix=np.round(M,2).tolist(),recall=np.round(rec,1).tolist(),precision=np.round(prec,1).tolist(),f1=np.round(f1,1).tolist(),accuracy=round(float(np.trace(M)),1),balanced_accuracy=round(float(np.mean(rec)),1),macro_f1=round(float(np.mean(f1)),1))
CM={}
for k in ['Multinomial logistic regression','LightGBM (tuned)','Ensemble (mean of LR, LightGBM, XGBoost, CatBoost)']:
    pi,pe=P[k]; CM[k]={'internal_argmax':cm(y[te],pi.argmax(1),w[te]),'internal_thresholds':cm(y[te],assign_thr(pi),w[te]),'external_argmax':cm(ya,pe.argmax(1),wa),'external_thresholds':cm(ya,assign_thr(pe),wa)}
    # Macro-F1 thresholds are NOT selected here: selecting them on the held-out set would leak evaluation data into the rule.
    # They are selected on out-of-fold development predictions in script 13 and only applied to the held-out and 2024 sets there.
log('CM',json.dumps(CM['Multinomial logistic regression']['internal_thresholds']))
# ---------- SHAP (tuned LightGBM, 19 değişken) ----------
idx=np.random.RandomState(0).choice(te,6000,replace=False); ex=shap.TreeExplainer(g); sv=ex.shap_values(XL.iloc[idx])
sv=np.array(sv);
if sv.shape[0]==3: sv=np.transpose(sv,(1,2,0))
imp=np.abs(sv).mean(0)  # (feat,3)
np.save('out/hoca_shap.npy',sv); XL.iloc[idx].to_pickle('out/hoca_shap_X.pkl')
shap_imp=sorted([[TOOL[i]]+[float(x) for x in imp[i]]+[float(imp[i].sum())] for i in range(len(TOOL))],key=lambda r:-r[4])
# LR göreli önem: her değişken için kategori log-odds aralığı (never ve overdue, referans güncel), ağırlıklı SD ile
coef=lr.coef_; groups={}
for j,c in enumerate(cols): groups.setdefault('yas' if c.startswith('yas') else c.split('=')[0],[]).append(j)
lr_imp=[]
for gk,js in groups.items():
    contrib=Z[te][:,js]@(coef[:,js].T)  # (n,3)
    d=contrib-contrib[:,[0]]; lr_imp.append([gk,float(np.sqrt(np.average(d[:,1]**2,weights=w[te])-np.average(d[:,1],weights=w[te])**2)),float(np.sqrt(np.average(d[:,2]**2,weights=w[te])-np.average(d[:,2],weights=w[te])**2))])
lr_imp.sort(key=lambda r:-(r[1]+r[2]))
# odds ratios (never vs current, overdue vs current) from the deployed model (refitted to all of 2020)
coefF=lrF.coef_; OR={}
for gk,js in groups.items():
    if gk in('yas','cocuk'): continue
    OR[gk]=[[cols[j].split('=')[1],float(np.exp(coefF[1,j]-coefF[0,j])),float(np.exp(coefF[2,j]-coefF[0,j]))] for j in js]
# LightGBM gain importance
gain=g.feature_importance('gain'); gain=gain/gain.sum()*100
json.dump(dict(training_note='held-out predictions: models fitted to the 80% development subset; 2024 and deployment predictions: same hyperparameters refitted to the full 2020 sample',holdout=HO,cm=CM,shap_importance=shap_imp,lr_importance=lr_imp,odds_ratios=OR,lgb_gain={TOOL[i]:float(gain[i]) for i in range(len(TOOL))},best_params={k:v[1] for k,v in best.items()},cv_best=CVbest,n_te=int(len(te)),n_tr=int(len(tr))),open('out/hoca.json','w'),indent=1)
g.save_model('out/hoca_lgb.txt'); log('BITTI')
