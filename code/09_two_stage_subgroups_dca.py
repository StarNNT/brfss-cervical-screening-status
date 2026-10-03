# Two-stage model, subgroup performance, group-specific thresholds, conformal prediction sets and decision curves. Run after script 03.
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'code')
from harmon import feats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
U='./'
b=pd.read_pickle('out/serviks2020.pkl').reset_index(drop=True); y=b.y.astype(int).values; wr=b._LLCPWT.values; w=wr/wr.mean()
a=pd.read_pickle('out/analitik.pkl').reset_index(drop=True); ya=a.y.map({'current':0,'overdue':1,'never':2}).values; wa=(a._LLCPWT/a._LLCPWT.mean()).values
XF=feats(b,2020); XA=feats(a,2024)
src=open('code/03_final_model_validation_export.py').read(); s0=src.index("V={'irk'"); s1=src.index("Z=design(XF)"); exec(src[s0:s1])
Z=design(XF); ZA=design(XA)
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
rs=np.random.RandomState(7)
def wauc(t,p,ww): return float(roc_auc_score(t,p,sample_weight=ww)) if 0<t.sum()<len(t) else np.nan
def boot_diff(t3,p1,p2,ww,fn,B=300):
    n=len(t3); d=[]
    for _ in range(B):
        i=rs.randint(0,n,n); d.append(fn(t3[i],p1[i],ww[i])-fn(t3[i],p2[i],ww[i]))
    return [round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)]
out={}
# 1. Two-stage model
import os
flat=LogisticRegression(C=1.0,max_iter=3000).fit(Z.iloc[tr],y[tr],sample_weight=w[tr]); pf=flat.predict_proba(Z.iloc[te])
sA=LogisticRegression(C=1.0,max_iter=3000).fit(Z.iloc[tr],(y[tr]==2).astype(int),sample_weight=w[tr])
scr=tr[y[tr]!=2]; sB=LogisticRegression(C=1.0,max_iter=3000).fit(Z.iloc[scr],(y[scr]==1).astype(int),sample_weight=w[scr])
def two_stage(Zx):
    pn=sA.predict_proba(Zx)[:,1]; po=sB.predict_proba(Zx)[:,1]; return np.c_[(1-pn)*(1-po),(1-pn)*po,pn]
ph=two_stage(Z.iloc[te])
flat_all=LogisticRegression(C=1.0,max_iter=3000).fit(Z,y,sample_weight=w); pfa=flat_all.predict_proba(ZA)
sA2=LogisticRegression(C=1.0,max_iter=3000).fit(Z,(y==2).astype(int),sample_weight=w); scr2=np.where(y!=2)[0]
sB2=LogisticRegression(C=1.0,max_iter=3000).fit(Z.iloc[scr2],(y[scr2]==1).astype(int),sample_weight=w[scr2])
pn=sA2.predict_proba(ZA)[:,1]; po=sB2.predict_proba(ZA)[:,1]; pha=np.c_[(1-pn)*(1-po),(1-pn)*po,pn]
def summ(t,p,ww):
    ev=t!=2; r={'auc_current':wauc(t==0,p[:,0],ww),'auc_overdue':wauc(t==1,p[:,1],ww),'auc_never':wauc(t==2,p[:,2],ww),'logloss':float(log_loss(t,p,sample_weight=ww)),
       'auc_overdue_among_screened':wauc(t[ev]==1,p[ev,1]/(p[ev,0]+p[ev,1]),ww[ev]),'mean_pred':[float(x) for x in np.average(p,axis=0,weights=ww)]}
    return {k:(round(v,4) if isinstance(v,float) else [round(x,4) for x in v]) for k,v in r.items()}
cmp={'internal':{'flat':summ(y[te],pf,w[te]),'two_stage':summ(y[te],ph,w[te])},'external_2024':{'flat':summ(ya,pfa,wa),'two_stage':summ(ya,pha,wa)}}
for nm,(t3,p1,p2,ww) in {'internal':(y[te],ph,pf,w[te]),'external_2024':(ya,pha,pfa,wa)}.items():
    cmp[nm]['diff_ci_two_minus_flat']={'auc_never':boot_diff(t3,p1,p2,ww,lambda t,p,q:wauc(t==2,p[:,2],q)),'auc_overdue':boot_diff(t3,p1,p2,ww,lambda t,p,q:wauc(t==1,p[:,1],q)),
        'logloss':boot_diff(t3,p1,p2,ww,lambda t,p,q:float(log_loss(t,p,sample_weight=q,labels=[0,1,2])))}
out['two_stage']=cmp; print(json.dumps(cmp,indent=1)); np.savez('out/genis_cache.npz',pf=pf,ph=ph,pfa=pfa,pha=pha)
# Stage B alone: overdue classification among previously screened participants
teS=te[y[te]!=2]; out['two_stage']['stageB_auc_internal']=round(wauc(y[teS]==1,sB.predict_proba(Z.iloc[teS])[:,1],w[teS]),4)
# 2. Subgroup performance
def groups(X):
    inc=X.gelir8
    return {'White, non-Hispanic':X.irk==1,'Black, non-Hispanic':X.irk==2,'Hispanic':X.irk==8,'Asian':X.irk==4,'AI/AN':X.irk==3,'Multiracial/other':X.irk.isin([5,6,7]),
     'Age 21-29':X.yas<30,'Age 30-49':(X.yas>=30)&(X.yas<50),'Age 50-65':X.yas>=50,'Insured':X.sigortasiz==0,'Uninsured':X.sigortasiz==1,
     'Urban':X.kirsal==0,'Rural':X.kirsal==1,'English interview':X.dil_ispanyolca==0,'Spanish interview':X.dil_ispanyolca==1,
     'Income <$25k':inc<=4,'Income $25k-$75k':(inc>=5)&(inc<=7),'Income >=$75k':inc==8,'Income unknown':inc.isna(),
     'No regular doctor':X.doktor==3,'Cost barrier':X.maliyet_engeli==1,'Less than high school':X.egitim<=3,'College graduate':X.egitim==6}
TH={'never':0.10,'overdue':0.15}
def fair(X,t,p,ww,label):
    rows=[]
    for g,m in ({'All':pd.Series(True,index=X.index)}|groups(X)).items():
        m=m.fillna(False).values
        if m.sum()<200: continue
        r={'group':g,'n':int(m.sum()),'obs_never':round(float(np.average(t[m]==2,weights=ww[m])*100),1),'obs_overdue':round(float(np.average(t[m]==1,weights=ww[m])*100),1),
           'pred_never':round(float(np.average(p[m,2],weights=ww[m])*100),1),'pred_overdue':round(float(np.average(p[m,1],weights=ww[m])*100),1),
           'auc_never':round(wauc(t[m]==2,p[m,2],ww[m]),3),'auc_overdue':round(wauc(t[m]==1,p[m,1],ww[m]),3)}
        for k,nm in [(2,'never'),(1,'overdue')]:
            f=p[m,k]>=TH[nm]; pos=(t[m]==k); W=ww[m]
            r[f'{nm}_flag']=round(float(np.average(f,weights=W)*100),1); r[f'{nm}_sens']=round(float(np.average(f[pos],weights=W[pos])*100),1) if pos.sum()>0 else None
            r[f'{nm}_spec']=round(float(np.average(~f[~pos],weights=W[~pos])*100),1); r[f'{nm}_ppv']=round(float(np.average(pos[f],weights=W[f])*100),1) if f.sum()>0 else None
        # Calibration slope on the logit scale for never-screened probability
        try:
            lg=np.log(p[m,2]/(1-p[m,2])); cs=LogisticRegression(C=1e6,max_iter=1000).fit(lg.reshape(-1,1),(t[m]==2).astype(int),sample_weight=ww[m]); r['cal_slope_never']=round(float(cs.coef_[0][0]),2)
        except Exception: r['cal_slope_never']=None
        rows.append(r)
    return rows
out['fairness']={'internal':fair(XF.iloc[te].reset_index(drop=True),y[te],pf,w[te],'internal'),'external_2024':fair(XA,ya,pfa,wa,'external')}
print(pd.DataFrame(out['fairness']['internal']).to_string()); print(pd.DataFrame(out['fairness']['external_2024']).to_string())
# Group-specific thresholds targeting 70% sensitivity in the held-out set
def thr_for_sens(t,p,ww,target=0.70):
    o=np.argsort(-p); pos=(t[o]==1)*ww[o]; cs=np.cumsum(pos)/pos.sum(); k=np.searchsorted(cs,target); return float(p[o][min(k,len(o)-1)])
gt=[]
Xt=XF.iloc[te].reset_index(drop=True)
for g,m in groups(Xt).items():
    m=m.fillna(False).values
    if m.sum()<500 or (y[te][m]==2).sum()<30: continue
    th=thr_for_sens((y[te][m]==2).astype(int),pf[m,2],w[te][m]); f=pf[m,2]>=th
    gt.append({'group':g,'threshold_never_for_70pct_sens':round(th*100,1),'flag_rate_at_group_threshold':round(float(np.average(f,weights=w[te][m])*100),1),
               'ppv_at_group_threshold':round(float(np.average((y[te][m]==2)[f],weights=w[te][m][f])*100),1)})
out['group_thresholds']=gt; print(pd.DataFrame(gt).to_string())
# 3. Conformal prediction sets
tr2,cal=train_test_split(tr,test_size=0.15,stratify=y[tr],random_state=3)
m2=LogisticRegression(C=1.0,max_iter=3000).fit(Z.iloc[tr2],y[tr2],sample_weight=w[tr2])
pc=m2.predict_proba(Z.iloc[cal]); pt=m2.predict_proba(Z.iloc[te]); pe=m2.predict_proba(ZA)
def wquant(v,ww,q):
    o=np.argsort(v); c=np.cumsum(ww[o])/ww.sum(); return float(v[o][np.searchsorted(c,q)])
def aps_score(p,lab):  # adaptive prediction sets (randomised değil)
    o=np.argsort(-p,axis=1); ps=np.take_along_axis(p,o,1); cs=np.cumsum(ps,1); rank=np.argmax(o==lab[:,None],axis=1); return cs[np.arange(len(p)),rank]
def aps_sets(p,q):
    o=np.argsort(-p,axis=1); ps=np.take_along_axis(p,o,1); cs=np.cumsum(ps,1); inc=cs-ps<q  # bir önceki kümülatif < q ise dahil
    S=np.zeros_like(p,bool); np.put_along_axis(S,o,inc,1); return S
conf={}
for alpha in (0.10,0.20):
    q=wquant(aps_score(pc,y[cal]),w[cal],1-alpha)
    for nm,(P,T,W,X) in {'internal':(pt,y[te],w[te],Xt),'external_2024':(pe,ya,wa,XA)}.items():
        S=aps_sets(P,q); cov=S[np.arange(len(T)),T]; sz=S.sum(1)
        r={'coverage':round(float(np.average(cov,weights=W))*100,1),'mean_set_size':round(float(np.average(sz,weights=W)),2),
           'singleton_pct':round(float(np.average(sz==1,weights=W))*100,1),'full_set_pct':round(float(np.average(sz==3,weights=W))*100,1),
           'coverage_by_class':{c:round(float(np.average(cov[T==k],weights=W[T==k]))*100,1) for k,c in enumerate(['current','overdue','never'])},
           'set_size_by_class':{c:round(float(np.average(sz[T==k],weights=W[T==k])),2) for k,c in enumerate(['current','overdue','never'])},
           'never_in_set_pct':round(float(np.average(S[:,2],weights=W))*100,1),'never_in_set_ppv':round(float(np.average((T==2)[S[:,2]],weights=W[S[:,2]]))*100,1),
           'never_in_set_sens':round(float(np.average(S[:,2][T==2],weights=W[T==2]))*100,1)}
        r['coverage_by_group']={g:round(float(np.average(cov[mm],weights=W[mm]))*100,1) for g,m in groups(X).items() if (mm:=m.fillna(False).values).sum()>=200}
        conf[f'alpha_{alpha}_{nm}']=r
out['conformal']=conf; print(json.dumps(conf,indent=1))
# 4. Decision-curve analysis for never-screened status
dca=[]
for nm,(P,T,W) in {'internal':(pf,y[te],w[te]),'external_2024':(pfa,ya,wa)}.items():
    prev=float(np.average(T==2,weights=W))
    for pt_ in [0.03,0.05,0.075,0.10,0.15,0.20,0.25,0.30]:
        f=P[:,2]>=pt_; tp=float(np.average(f&(T==2),weights=W)); fp=float(np.average(f&(T!=2),weights=W))
        nb=tp-fp*pt_/(1-pt_); nb_all=prev-(1-prev)*pt_/(1-pt_)
        dca.append({'set':nm,'threshold':pt_,'net_benefit_model':round(nb*100,2),'net_benefit_contact_all':round(nb_all*100,2),'flag_rate':round(float(np.average(f,weights=W))*100,1)})
out['decision_curve']=dca; print(pd.DataFrame(dca).to_string())
json.dump(out,open('out/genisletme.json','w'),ensure_ascii=False,indent=1); print('BITTI')
