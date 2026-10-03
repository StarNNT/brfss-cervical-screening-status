# Design-based (within-stratum) bootstrap with 1,000 resamples for all reported metrics and differences (multiplicities via np.add.at); thresholds selected on out-of-fold development predictions; state-grouped 10-fold CV; unweighted split conformal with finite-sample correction, Wilson CIs for coverage and empty-set accounting; fold SDs for tuned models. Run after 11. Usage: python code/13_...py [B]
import pandas as pd, numpy as np, sys, json, warnings, time; warnings.filterwarnings('ignore'); sys.path.insert(0,'code')
from harmon import feats, NUM
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
LOG=open('out/design_bootstrap.log','a')
def log(*x): print(*x,flush=True); LOG.write(' '.join(map(str,x))+'\n'); LOG.flush()
OUT={}
def save(): json.dump(OUT,open('out/design_bootstrap.json','w'),indent=1)
b=pd.read_pickle('out/serviks2020.pkl').reset_index(drop=True); y=b.y.astype(int).values; w=(b._LLCPWT/b._LLCPWT.mean()).values; st=b._STSTR.values; state=b._STATE.values
a=pd.read_pickle('out/analitik.pkl').reset_index(drop=True); ya=a.y.map({'current':0,'overdue':1,'never':2}).values; wa=(a._LLCPWT/a._LLCPWT.mean()).values; sta=a._STSTR.values
TOOL=['yas','cocuk','irk','egitim','gelir8','medeni','istihdam','ev_sahibi','kirsal','dil_ispanyolca','sigortasiz','doktor','maliyet_engeli','checkup','genel_saglik','sigara','grip_asisi','hiv_testi','dis_hekimi']
XF=feats(b,2020); XA=feats(a,2024); X=XF[TOOL]; X24=XA[TOOL]
src=open('code/03_final_model_validation_export.py').read(); exec(src[src.index("V={'irk'"):src.index("Z=design(XF)")])
Zd=design(XF); cols=list(Zd.columns); Z=Zd.values.astype('float32'); Z24=design(XA)[cols].values.astype('float32')
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
Pz=np.load('out/model_predictions.npz'); P_te={k:Pz[f'{i}_te'] for i,k in enumerate(['lr','lgb','xgb','cb','ens4','ens3'])}; P_24={k:Pz[f'{i}_24'] for i,k in enumerate(['lr','lgb','xgb','cb','ens4','ens3'])}
Gc=np.load('out/genis_cache.npz'); P_te['two']=Gc['ph']; P_24['two']=Gc['pha']
assert np.allclose(P_te['lr'],Gc['pf'],atol=1e-6)
# Fast weighted AUC for fixed scores and resampled survey weights
class FastAUC:
    def __init__(s,score,lab):
        o=np.argsort(score,kind='stable'); s.o=o; ss=score[o]; s.lab=lab[o].astype(float)
        brk=np.r_[True,ss[1:]!=ss[:-1]]; s.starts=np.flatnonzero(brk); s.gid=np.cumsum(brk)-1
    def __call__(s,v):  # v: ağırlık vektörü (orijinal sırada)
        v=v[s.o]; pos=v*s.lab; neg=v*(1-s.lab)
        gp=np.add.reduceat(pos,s.starts); gn=np.add.reduceat(neg,s.starts)
        cneg=np.cumsum(gn)-gn  # gruptan önceki negatif kütle
        P=gp.sum(); N=gn.sum()
        if P<=0 or N<=0: return np.nan
        return float((gp*(cneg+0.5*gn)).sum()/(P*N))
def wlogloss(t,p,v): return float(-(v*np.log(np.clip(p[np.arange(len(t)),t],1e-12,1))).sum()/v.sum())
def brier(t,pk,k,v): return float((v*(pk-(t==k))**2).sum()/v.sum())
def calib_ab(t,pk,k,v,it=25):  # ağırlıklı lojistik yeniden kalibrasyon: logit(p)=a+b*lp (Newton)
    lp=np.log(np.clip(pk,1e-9,1-1e-9)/(1-np.clip(pk,1e-9,1-1e-9))); yy=(t==k).astype(float); a,bb=0.0,1.0
    for _ in range(it):
        z=a+bb*lp; p=1/(1+np.exp(-z)); r=v*(yy-p); W=v*p*(1-p)
        g=np.array([r.sum(),(r*lp).sum()]); H=np.array([[W.sum(),(W*lp).sum()],[(W*lp).sum(),(W*lp*lp).sum()]])
        try: d=np.linalg.solve(H+1e-9*np.eye(2),g)
        except Exception: break
        a+=d[0]; bb+=d[1]
        if np.abs(d).max()<1e-8: break
    return float(a),float(bb)
def ece(t,pk,k,v,edges):
    yy=(t==k).astype(float); bi=np.clip(np.searchsorted(edges,pk,side='right')-1,0,len(edges)-2); e=0.0
    for j in range(len(edges)-1):
        m=bi==j; vw=v[m].sum()
        if vw>0: e+=vw/v.sum()*abs(np.average(pk[m],weights=v[m])-np.average(yy[m],weights=v[m]))
    return float(e)
def strat_idx(strata):
    u,inv=np.unique(strata,return_inverse=True); return [np.flatnonzero(inv==i) for i in range(len(u))]
def boot_mult(groups,rs,n):
    # Resample with replacement within strata. np.add.at preserves repeated draws.
    m=np.zeros(n)
    for g in groups:
        k=len(g); np.add.at(m,g[rs.randint(0,k,k)],1)
    return m
def assign_thr(p,tn,to): return np.where(p[:,2]>=tn,2,np.where(p[:,1]>=to,1,0))
def netben(t,p,v,k,pt):
    f=p[:,k]>=pt; W=v.sum(); TP=v[f&(t==k)].sum()/W; FP=v[f&(t!=k)].sum()/W; prev=v[t==k].sum()/W
    return (TP-FP*pt/(1-pt))*100,(prev-(1-prev)*pt/(1-pt))*100
def groups(X):
    inc=X.gelir8
    return {'White, non-Hispanic':X.irk==1,'Black, non-Hispanic':X.irk==2,'Hispanic':X.irk==8,'Asian':X.irk==4,'AI/AN':X.irk==3,'Multiracial/other':X.irk.isin([5,6,7]),
     'Age 21-29':X.yas<30,'Age 30-49':(X.yas>=30)&(X.yas<50),'Age 50-65':X.yas>=50,'Insured':X.sigortasiz==0,'Uninsured':X.sigortasiz==1,'Urban':X.kirsal==0,'Rural':X.kirsal==1,
     'English interview':X.dil_ispanyolca==0,'Spanish interview':X.dil_ispanyolca==1,'Income <$25k':inc<=4,'Income $25k-$75k':(inc>=5)&(inc<=7),'Income >=$75k':inc==8,'Income unknown':inc.isna(),
     'No regular doctor':X.doktor==3,'Cost barrier':X.maliyet_engeli==1,'Less than high school':X.egitim<=3,'College graduate':X.egitim==6}
CL=['current','overdue','never']
# 1. Out-of-fold logistic-regression predictions and thresholds
import os
t0=time.time(); oof=np.zeros((len(tr),3)); folds=list(StratifiedKFold(5,shuffle=True,random_state=7).split(Z[tr],y[tr]))
if os.path.exists('out/oof_predictions.npy'): oof=np.load('out/oof_predictions.npy')
else:
    for itr,ite in folds:
        m=LogisticRegression(C=1.0,max_iter=3000).fit(Z[tr[itr]],y[tr[itr]],sample_weight=w[tr[itr]]); oof[ite]=m.predict_proba(Z[tr[ite]])
    np.save('out/oof_predictions.npy',oof); log('OOF done',time.time()-t0)
ytr,wtr=y[tr],w[tr]
def cm_stats(t,pred,v):
    M=np.zeros((3,3))
    for i in range(3):
        for j in range(3): M[i,j]=v[(t==i)&(pred==j)].sum()
    M=M/v.sum()*100; rec=[M[i,i]/M[i].sum()*100 for i in range(3)]; prec=[M[i,i]/M[:,i].sum()*100 if M[:,i].sum()>0 else 0 for i in range(3)]
    f1=[2*r*p/(r+p) if r+p>0 else 0 for r,p in zip(rec,prec)]
    return dict(matrix=np.round(M,2).tolist(),recall=np.round(rec,1).tolist(),precision=np.round(prec,1).tolist(),f1=np.round(f1,1).tolist(),accuracy=round(float(np.trace(M)),1),balanced_accuracy=round(float(np.mean(rec)),1),macro_f1=round(float(np.mean(f1)),1))
grid=[(tn,to) for tn in np.arange(0.04,0.42,0.02) for to in np.arange(0.06,0.52,0.02)]
bestf=max(((tn,to,cm_stats(ytr,assign_thr(oof,tn,to),wtr)['macro_f1']) for tn,to in grid),key=lambda r:r[2])
OUT['thresholds']={'macro_f1_selected_on_oof':{'never':round(float(bestf[0]),2),'overdue':round(float(bestf[1]),2),'oof_macro_f1':bestf[2]},'default':{'never':0.10,'overdue':0.15}}
# Group-specific thresholds targeting 70% sensitivity
Xtr=XF.iloc[tr].reset_index(drop=True); gt={}
def thr_for_sens(t,p,v,target=0.70):
    o=np.argsort(-p); pos=(t[o]==1)*v[o]; cs=np.cumsum(pos)/pos.sum(); k=np.searchsorted(cs,target); return float(p[o][min(k,len(o)-1)])
for g,m in groups(Xtr).items():
    m=m.fillna(False).values
    if m.sum()<500 or (ytr[m]==2).sum()<30: continue
    gt[g]=round(thr_for_sens((ytr[m]==2).astype(int),oof[m,2],wtr[m])*100,1)
OUT['thresholds']['group_70pct_sens_selected_on_oof']=gt
# Apply the selected thresholds to the held-out 2020 and 2024 samples
Xt=XF.iloc[te].reset_index(drop=True)
OUT['cm']={}
for nm,(t,P,v) in {'internal':(y[te],P_te,w[te]),'external_2024':(ya,P_24,wa)}.items():
    OUT['cm'][nm]={}
    for mk in ['lr','lgb','ens4']:
        OUT['cm'][nm][mk]={'argmax':cm_stats(t,P[mk].argmax(1),v),'default_10_15':cm_stats(t,assign_thr(P[mk],0.10,0.15),v),'macro_f1_oof':cm_stats(t,assign_thr(P[mk],bestf[0],bestf[1]),v)}
save(); log('thresholds done')
# 2. Design-based bootstrap
B=int(sys.argv[1]) if len(sys.argv)>1 else 1000
def run_boot(nm,t,P,v,strata,Xg,rs):
    n=len(t); grp=strat_idx(strata); G={g:m.fillna(False).values for g,m in groups(Xg).items()}
    A={};
    for mk in ['lr','lgb','xgb','cb','ens4','ens3','two']:
        for k in range(3): A[(mk,k)]=FastAUC(P[mk][:,k],(t==k).astype(int))
    ev=t!=2; condp=P['lr'][:,1]/(P['lr'][:,0]+P['lr'][:,1]); Acond=FastAUC(np.where(ev,condp,-1),(t==1).astype(int))  # ev dışı v=0 ile
    edges={k:np.quantile(P['lr'][:,k],np.linspace(0,1,11)) for k in range(3)}; edges={k:np.r_[-1,e[1:-1],2] for k,e in edges.items()}
    gthr=OUT['thresholds']['group_70pct_sens_selected_on_oof']
    def stats(m):
        vv=v*m; r={}
        for mk in ['lr','lgb','xgb','cb','ens4','ens3','two']:
            for k in range(3): r[f'auc_{mk}_{CL[k]}']=A[(mk,k)](vv)
            r[f'auc_{mk}_macro']=np.mean([r[f'auc_{mk}_{c}'] for c in CL]); r[f'logloss_{mk}']=wlogloss(t,P[mk],vv)
        r['auc_lr_overdue_among_screened']=Acond(vv*ev)
        for k in range(3):
            c=CL[k]; r[f'brier_lr_{c}']=brier(t,P['lr'][:,k],k,vv); aa,bb=calib_ab(t,P['lr'][:,k],k,vv); r[f'cal_int_lr_{c}']=aa; r[f'cal_slope_lr_{c}']=bb; r[f'ece_lr_{c}']=ece(t,P['lr'][:,k],k,vv,edges[k])
            r[f'mean_pred_lr_{c}']=float(np.average(P['lr'][:,k],weights=vv)); r[f'obs_{c}']=float(np.average(t==k,weights=vv))
        for k,c,thr in [(2,'never',0.10),(1,'overdue',0.15)]:
            f=P['lr'][:,k]>=thr; pos=t==k; TP=vv[f&pos].sum(); FP=vv[f&~pos].sum(); FN=vv[~f&pos].sum(); TN=vv[~f&~pos].sum()
            r[f'alert_{c}']=(TP+FP)/vv.sum()*100; r[f'sens_{c}']=TP/(TP+FN)*100; r[f'spec_{c}']=TN/(TN+FP)*100; r[f'ppv_{c}']=TP/(TP+FP)*100; r[f'npv_{c}']=TN/(TN+FN)*100; r[f'fpr_{c}']=FP/(FP+TN)*100
            for pt in [0.05,0.10,0.15,0.20,0.30]:
                nb,nba=netben(t,P['lr'],vv,k,pt); r[f'nb_{c}_{pt}']=nb; r[f'nball_{c}_{pt}']=nba
        # Lift in the top 20% of predicted probabilities
        o=np.argsort(-P['lr'][:,2]); cw=np.cumsum(vv[o])/vv.sum(); tt=(t[o]==2)*vv[o]; r['lift20_never']=tt[cw<=0.2].sum()/tt.sum()*100
        # Subgroup metrics
        for g,m2 in G.items():
            vg=vv*m2
            if (vg>0).sum()<200 or ((t==2)&(vg>0)).sum()<10: continue
            r[f'g_auc_never_{g}']=A[('lr',2)](vg); r[f'g_auc_overdue_{g}']=A[('lr',1)](vg)
            f=P['lr'][:,2]>=0.10; pos=t==2; TP=vg[f&pos].sum(); FP=vg[f&~pos].sum(); FN=vg[~f&pos].sum(); TN=vg[~f&~pos].sum()
            r[f'g_sens_{g}']=TP/(TP+FN)*100 if TP+FN>0 else np.nan; r[f'g_ppv_{g}']=TP/(TP+FP)*100 if TP+FP>0 else np.nan; r[f'g_fpr_{g}']=FP/(FP+TN)*100; r[f'g_alert_{g}']=(TP+FP)/vg.sum()*100
            r[f'g_obs_never_{g}']=float(np.average(t==2,weights=vg)); r[f'g_pred_never_{g}']=float(np.average(P['lr'][:,2],weights=vg)); r[f'g_obs_overdue_{g}']=float(np.average(t==1,weights=vg)); r[f'g_pred_overdue_{g}']=float(np.average(P['lr'][:,1],weights=vg))
            if g in gthr:
                f2=P['lr'][:,2]>=gthr[g]/100; TP=vg[f2&pos].sum(); FP=vg[f2&~pos].sum(); FN=vg[~f2&pos].sum(); r[f'g_sens_gthr_{g}']=TP/(TP+FN)*100; r[f'g_alert_gthr_{g}']=(TP+FP)/vg.sum()*100; r[f'g_ppv_gthr_{g}']=TP/(TP+FP)*100 if TP+FP>0 else np.nan
        return r
    point=stats(np.ones(n)); keys=list(point); R=np.full((B,len(keys)),np.nan); t0=time.time()
    for bi in range(B):
        m=boot_mult(grp,rs,n); s=stats(m); R[bi]=[s[k] for k in keys]
        if bi%100==0: log(nm,'boot',bi,round(time.time()-t0))
    res={}
    for j,k in enumerate(keys):
        lo,hi=np.nanpercentile(R[:,j],[2.5,97.5]); res[k]={'est':point[k],'lo':float(lo),'hi':float(hi)}
    # Paired differences within the same resample: model minus logistic regression
    for mk in ['lgb','xgb','cb','ens4','ens3','two']:
        for met in ['auc_never','auc_overdue','auc_current','auc_macro']:
            c=met.replace('auc_',''); a1=keys.index(f'auc_{mk}_{c}'); a0=keys.index(f'auc_lr_{c}'); d=R[:,a1]-R[:,a0]
            res[f'diff_{mk}_{met}']={'est':point[f'auc_{mk}_{c}']-point[f'auc_lr_{c}'],'lo':float(np.nanpercentile(d,2.5)),'hi':float(np.nanpercentile(d,97.5))}
        a1=keys.index(f'logloss_{mk}'); a0=keys.index('logloss_lr'); d=R[:,a1]-R[:,a0]; res[f'diff_{mk}_logloss']={'est':point[f'logloss_{mk}']-point['logloss_lr'],'lo':float(np.nanpercentile(d,2.5)),'hi':float(np.nanpercentile(d,97.5))}
    for c in ['never','overdue']:
        for pt in [0.05,0.10,0.15,0.20,0.30]:
            a1=keys.index(f'nb_{c}_{pt}'); a0=keys.index(f'nball_{c}_{pt}'); d=R[:,a1]-R[:,a0]; res[f'nbdiff_{c}_{pt}']={'est':point[f'nb_{c}_{pt}']-point[f'nball_{c}_{pt}'],'lo':float(np.nanpercentile(d,2.5)),'hi':float(np.nanpercentile(d,97.5))}
    res['_B']=B; res['_n_strata']=len(grp); return res
rs=np.random.RandomState(2026)
OUT['boot_internal']=run_boot('internal',y[te],P_te,w[te],st[te],Xt,rs); save(); log('boot internal done')
OUT['boot_external_2024']=run_boot('external',ya,P_24,wa,sta,XA,rs); save(); log('boot external done')
# 3. Unweighted split conformal prediction with finite-sample correction
tr2,cal=train_test_split(tr,test_size=0.15,stratify=y[tr],random_state=3)
m2=LogisticRegression(C=1.0,max_iter=3000).fit(Z[tr2],y[tr2],sample_weight=w[tr2]); pc=m2.predict_proba(Z[cal]); pt_=m2.predict_proba(Z[te]); pe=m2.predict_proba(Z24)
def qhat(scores,alpha):
    n=len(scores); k=int(np.ceil((n+1)*(1-alpha))); k=min(k,n); return float(np.sort(scores)[k-1])
def wilson(x,n,z=1.96):
    if n==0: return [np.nan,np.nan]
    p=x/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [round(float(c-h)*100,1),round(float(c+h)*100,1)]
def evalset(S,t,v,P=None):
    cov=S[np.arange(len(t)),t]; sz=S.sum(1)
    r={'coverage_unweighted_ci':wilson(int(cov.sum()),len(t)),'coverage_by_class_unweighted_ci':{c:wilson(int(cov[t==k].sum()),int((t==k).sum())) for k,c in enumerate(CL)},'n_by_class':{c:int((t==k).sum()) for k,c in enumerate(CL)},
       'coverage_unweighted':round(float(cov.mean())*100,1),'coverage_weighted':round(float(np.average(cov,weights=v))*100,1),'mean_set_size':round(float(sz.mean()),2),'singleton_pct':round(float((sz==1).mean())*100,1),'full_pct':round(float((sz==3).mean())*100,1),'empty_pct':round(float((sz==0).mean())*100,1),
       'coverage_by_class_unweighted':{c:round(float(cov[t==k].mean())*100,1) for k,c in enumerate(CL)},'coverage_by_class_weighted':{c:round(float(np.average(cov[t==k],weights=v[t==k]))*100,1) for k,c in enumerate(CL)},
       'never_in_set_pct':round(float(S[:,2].mean())*100,1),'never_in_set_sens':round(float(S[t==2,2].mean())*100,1),'never_in_set_ppv':round(float((t[S[:,2]]==2).mean())*100,1),'set_overdue_never_only_pct':round(float(((sz==2)&S[:,1]&S[:,2]).mean())*100,1),'singleton_never_pct':round(float(((sz==1)&S[:,2]).mean())*100,1)}
    if P is not None:  # boş kümeler: hiçbir sınıf eşiği geçmemiş; işletimsel kural olarak en olası sınıf eklenir (kapsama yalnızca artar)
        S2=S.copy(); e=sz==0; S2[e,P[e].argmax(1)]=True; cov2=S2[np.arange(len(t)),t]
        r['empty_n']=int(e.sum()); r['empty_obs_class_pct']={c:round(float((t[e]==k).mean())*100,1) for k,c in enumerate(CL)} if e.sum()>0 else None
        r['coverage_unweighted_with_argmax_fallback']=round(float(cov2.mean())*100,1); r['coverage_by_class_unweighted_with_argmax_fallback']={c:round(float(cov2[t==k].mean())*100,1) for k,c in enumerate(CL)}
        r['mean_set_size_with_argmax_fallback']=round(float(S2.sum(1).mean()),2)
    return r
conf={}
for alpha in (0.10,0.20):
    s_cal=1-pc[np.arange(len(cal)),y[cal]]; q=qhat(s_cal,alpha); qk=[qhat(1-pc[y[cal]==k,k],alpha) for k in range(3)]
    conf[f'alpha_{alpha}']={'marginal_floor':round(1-q,4),'classwise_floor':{c:round(1-qk[k],4) for k,c in enumerate(CL)},'n_cal':int(len(cal)),'n_cal_by_class':{c:int((y[cal]==k).sum()) for k,c in enumerate(CL)}}
    for nm,(P,T,Wv) in {'internal':(pt_,y[te],w[te]),'external_2024':(pe,ya,wa)}.items():
        conf[f'alpha_{alpha}'][nm]={'marginal':evalset(P>=1-q,T,Wv,P),'classwise':evalset(np.stack([P[:,k]>=1-qk[k] for k in range(3)],1),T,Wv,P)}
        # Class-conditional subgroup coverage for never-screened status
        Xg=Xt if nm=='internal' else XA; S=np.stack([P[:,k]>=1-qk[k] for k in range(3)],1); cov=S[np.arange(len(T)),T]
        conf[f'alpha_{alpha}'][nm]['classwise_never_coverage_by_group']={g:round(float(cov[(mm)&(T==2)].mean())*100,1) for g,m in groups(Xg).items() if ((mm:=m.fillna(False).values)&(T==2)).sum()>=30}
OUT['conformal']=conf; save(); log('conformal done')
# 4. Ten-fold cross-validation grouped by state
XL=X.copy()
for f in XL.columns:
    if f not in NUM: XL[f]=XL[f].astype('category')
LGBP={'learning_rate': 0.02, 'num_leaves': 15, 'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.4, 'reg_lambda': 1.0, 'cat_smooth': 100.0, 'max_cat_to_onehot': 4}
def fit_lgb(itr,nround=600): return lgb.train(dict(objective='multiclass',num_class=3,verbose=-1,n_jobs=2,seed=42,subsample_freq=1)|LGBP,lgb.Dataset(XL.iloc[itr],y[itr],weight=w[itr]),nround)
gcv=[]; oofg=np.zeros((len(y),3)); oofl=np.zeros((len(y),3))
for k,(itr,ite) in enumerate(GroupKFold(10).split(Z,y,groups=state)):
    m=LogisticRegression(C=1.0,max_iter=3000).fit(Z[itr],y[itr],sample_weight=w[itr]); p=m.predict_proba(Z[ite]); oofg[ite]=p
    g=fit_lgb(itr); pl=g.predict(XL.iloc[ite]); oofl[ite]=pl
    def met(p): return {c:FastAUC(p[:,j],(y[ite]==j).astype(int))(w[ite]) for j,c in enumerate(CL)}|{'logloss':wlogloss(y[ite],p,w[ite])}
    r={'fold':k,'states':sorted(set(int(s) for s in state[ite])),'n':int(len(ite)),'lr':met(p),'lgb':met(pl)}; gcv.append(r); log('groupcv',k,r['lr'],r['lgb'])
pool=lambda p:{c:FastAUC(p[:,j],(y==j).astype(int))(w) for j,c in enumerate(CL)}
OUT['state_grouped_cv']={'folds':gcv,'pooled_lr':pool(oofg),'pooled_lgb':pool(oofl),'summary':{mk:{c:{'mean':float(np.mean([f[mk][c] for f in gcv])),'sd':float(np.std([f[mk][c] for f in gcv])),'min':float(np.min([f[mk][c] for f in gcv])),'max':float(np.max([f[mk][c] for f in gcv]))} for c in CL} for mk in ['lr','lgb']}}
save(); log('state grouped cv done')
# 5. Fold-level standard deviations for tuned models
XC=X.copy(); CAT=[c for c in TOOL if c not in NUM]
for c in CAT: XC[c]=XC[c].fillna(-1).astype(int).astype(str)
XGBP={'n_estimators': 800, 'learning_rate': 0.02, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_lambda': 20.0, 'min_child_weight': 1.0}; CBP={'iterations': 500, 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 1.0}
tuned={}
for mk in ['lgb','xgb','cb']:
    ms=[]
    for itr,ite in folds:
        I=tr[itr]; E=tr[ite]
        if mk=='lgb': p=fit_lgb(I).predict(XL.iloc[E])
        elif mk=='xgb': p=xgb.XGBClassifier(n_jobs=2,tree_method='hist',random_state=0,**XGBP).fit(Z[I],y[I],sample_weight=w[I]).predict_proba(Z[E])
        else: p=CatBoostClassifier(loss_function='MultiClass',verbose=0,thread_count=2,random_seed=0,cat_features=CAT,**CBP).fit(XC.iloc[I],y[I],sample_weight=w[I]).predict_proba(XC.iloc[E])
        r={c:FastAUC(p[:,j],(y[E]==j).astype(int))(w[E]) for j,c in enumerate(CL)}; r['macro']=np.mean(list(r.values())); r['logloss']=wlogloss(y[E],p,w[E]); ms.append(r)
    tuned[mk]={k_:{'mean':float(np.mean([m[k_] for m in ms])),'sd':float(np.std([m[k_] for m in ms]))} for k_ in ms[0]}; log('tuned cv',mk,tuned[mk]); OUT['tuned_cv_sd']=tuned; save()
log('BITTI')
