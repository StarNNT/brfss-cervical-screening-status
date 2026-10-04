# Random hyperparameter search (LightGBM 24, XGBoost 10, CatBoost 6 draws) and 5-fold CV of all algorithms in the development set.
# Slow (roughly 2–4 hours on 2 cores, depending on CatBoost speed). Writes out/search_results.json and out/search_best.json, which script 11 reads.
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
LOG=open('out/model_search.log','a');
def log(*x): print(*x,flush=True); LOG.write(' '.join(map(str,x))+'\n'); LOG.flush()
# Candidate models
def fit_lr(itr): return LogisticRegression(C=1.0,max_iter=3000).fit(Z[itr],y[itr],sample_weight=w[itr])
def fit_lgb(itr,par,nround):
    p=dict(objective='multiclass',num_class=3,verbose=-1,n_jobs=2,seed=42,subsample_freq=1)|par
    return lgb.train(p,lgb.Dataset(XL.iloc[itr],y[itr],weight=w[itr]),nround)
def fit_xgb(itr,par): return xgb.XGBClassifier(n_jobs=2,tree_method='hist',random_state=0,**par).fit(Z[itr],y[itr],sample_weight=w[itr])
def fit_cb(itr,par): return CatBoostClassifier(loss_function='MultiClass',verbose=0,thread_count=2,random_seed=0,cat_features=CAT,**par).fit(XC.iloc[itr],y[itr],sample_weight=w[itr])
LGB0=dict(learning_rate=0.05,num_leaves=31,min_child_samples=80,subsample=0.8,colsample_bytree=0.7,reg_lambda=5,cat_smooth=30); XGB0=dict(n_estimators=350,learning_rate=0.05,max_depth=5,subsample=0.8,colsample_bytree=0.7,reg_lambda=5)
folds=list(StratifiedKFold(5,shuffle=True,random_state=7).split(Z[tr],y[tr]))
# Random search scored by cross-validated log loss
rs=np.random.RandomState(11); search=[]
grid_lgb=[dict(learning_rate=float(rs.choice([0.02,0.03,0.05,0.08])),num_leaves=int(rs.choice([7,15,31,63,127])),min_child_samples=int(rs.choice([20,50,100,200,400])),subsample=float(rs.choice([0.6,0.8,1.0])),colsample_bytree=float(rs.choice([0.4,0.6,0.8,1.0])),reg_lambda=float(rs.choice([0,1,5,20,50])),cat_smooth=float(rs.choice([10,30,100])),max_cat_to_onehot=int(rs.choice([4,8])),nround=int(rs.choice([150,300,600,1000]))) for _ in range(24)]
grid_xgb=[dict(n_estimators=int(rs.choice([200,400,800])),learning_rate=float(rs.choice([0.02,0.05,0.1])),max_depth=int(rs.choice([3,4,6,8])),subsample=float(rs.choice([0.6,0.8,1.0])),colsample_bytree=float(rs.choice([0.5,0.7,1.0])),reg_lambda=float(rs.choice([1,5,20])),min_child_weight=float(rs.choice([1,5,20]))) for _ in range(10)]
grid_cb=[dict(iterations=int(rs.choice([500,1000])),learning_rate=float(rs.choice([0.03,0.06,0.1])),depth=int(rs.choice([4,6,8])),l2_leaf_reg=float(rs.choice([1,3,10]))) for _ in range(6)]
def cv_eval(fitter,predict,name,par):
    t0=time.time(); ms=[]; P=np.zeros((len(tr),3))
    for k,(itr,ite) in enumerate(folds):
        m=fitter(tr[itr],par); p=predict(m,tr[ite]); P[ite]=p; ms.append(metrics(y[tr[ite]],p,w[tr[ite]]))
    agg={k_:(float(np.mean([m[k_] for m in ms])),float(np.std([m[k_] for m in ms]))) for k_ in ms[0]}
    log(name,json.dumps(par),'macro %.4f logloss %.4f (%.0fs)'%(agg['auc_macro'][0],agg['logloss'][0],time.time()-t0)); return agg,P
pred_lgb=lambda m,i:m.predict(XL.iloc[i]); pred_xgb=lambda m,i:m.predict_proba(Z[i]); pred_cb=lambda m,i:m.predict_proba(XC.iloc[i]); pred_lr=lambda m,i:m.predict_proba(Z[i])
CV={}; OOF={}
CV['Multinomial logistic regression'],OOF['lr']=cv_eval(lambda i,p:fit_lr(i),pred_lr,'LR',{})
CV['Decision tree'],_=cv_eval(lambda i,p:DecisionTreeClassifier(max_depth=8,min_samples_leaf=200,random_state=0).fit(Z[i],y[i],sample_weight=w[i]),pred_lr,'DT',{})
CV['Random forest'],_=cv_eval(lambda i,p:RandomForestClassifier(n_estimators=300,min_samples_leaf=20,n_jobs=2,random_state=0).fit(Z[i],y[i],sample_weight=w[i]),pred_lr,'RF',{})
CV['Multilayer perceptron'],_=cv_eval(lambda i,p:MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-2,early_stopping=True,n_iter_no_change=5,max_iter=60,random_state=0).fit(Zs[i],y[i]),lambda m,i:m.predict_proba(Zs[i]),'MLP',{})
CV['XGBoost (default)'],_=cv_eval(fit_xgb,pred_xgb,'XGB0',XGB0)
CV['LightGBM (default)'],OOF['lgb0']=cv_eval(lambda i,p:fit_lgb(i,{k:v for k,v in p.items() if k!='nround'},180),pred_lgb,'LGB0',LGB0)
best={}
for nm,grid,fitter,pred in [('lgb',grid_lgb,lambda i,p:fit_lgb(i,{k:v for k,v in p.items() if k!='nround'},p['nround']),pred_lgb),('xgb',grid_xgb,fit_xgb,pred_xgb),('cb',grid_cb,fit_cb,pred_cb)]:
    res=[]
    for par in grid:
        agg,P=cv_eval(fitter,pred,nm.upper(),par); res.append((agg['logloss'][0],par,agg,P)); search.append(dict(model=nm,par=par,auc_macro=agg['auc_macro'][0],logloss=agg['logloss'][0]))
    res.sort(key=lambda r:r[0]); best[nm]=res[0]; OOF[nm]=res[0][3]
CV['LightGBM (tuned)']=best['lgb'][2]; CV['XGBoost (tuned)']=best['xgb'][2]; CV['CatBoost (tuned)']=best['cb'][2]
# Ensemble of out-of-fold probability estimates
ens=(OOF['lr']+OOF['lgb']+OOF['xgb']+OOF['cb'])/4; ms=[metrics(y[tr[ite]],ens[ite],w[tr[ite]]) for _,ite in folds]
CV['Ensemble (mean of LR, LightGBM, XGBoost, CatBoost)']={k_:(float(np.mean([m[k_] for m in ms])),float(np.std([m[k_] for m in ms]))) for k_ in ms[0]}
ens3=(OOF['lgb']+OOF['xgb']+OOF['cb'])/3; ms=[metrics(y[tr[ite]],ens3[ite],w[tr[ite]]) for _,ite in folds]
CV['Ensemble (mean of LightGBM, XGBoost, CatBoost)']={k_:(float(np.mean([m[k_] for m in ms])),float(np.std([m[k_] for m in ms]))) for k_ in ms[0]}
log('CV done'); json.dump(dict(cv={k:{m:list(v) for m,v in agg.items()} for k,agg in CV.items()},search=search,best={k:v[1] for k,v in best.items()}),open('out/search_results.json','w'),indent=1); json.dump({k:v[1] for k,v in best.items()},open('out/search_best.json','w'),indent=1)
