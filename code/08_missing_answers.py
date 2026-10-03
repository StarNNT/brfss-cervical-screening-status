# Eksik yanıtın iki ele alınış biçimi: (a) "bilinmiyor" kategorisi (anketteki gibi), (b) o sorunun nüfus ortalamasıyla doldurma
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'out'); sys.path.insert(0,'code')
from harmon import feats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
b=pd.read_pickle('out/serviks2020.pkl').reset_index(drop=True); y=b.y.astype(int).values; w=(b._LLCPWT/b._LLCPWT.mean()).values
X=feats(b,2020); J=json.load(open('out/webmodel.json')); M=J['model']; COLS=M['cols']; VARS=list(M['vars']); LEV={v:[c for c,_ in M['vars'][v]['levels']] for v in VARS}
def design(X):
    D={}
    for c in COLS:
        if c=='yas': D[c]=X.yas-21
        elif c.startswith('yas_k'): D[c]=np.clip(X.yas-int(c[5:]),0,None)
        elif c=='cocuk': D[c]=X.cocuk.fillna(0)
        else:
            v,l=c.split('='); D[c]=((X[v].isna()|~X[v].isin(LEV[v])) if l=='NA' else (X[v]==int(l))).astype(float)
    return pd.DataFrame(D)
Z=design(X); mu=(Z.values*w[:,None]).sum(0)/w.sum(); MU=dict(zip(COLS,mu))
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
lr=LogisticRegression(C=1.0,max_iter=3000).fit(Z.iloc[tr],y[tr],sample_weight=w[tr]); Xt=X.iloc[te].reset_index(drop=True); yt=y[te]; wt=w[te]; rs=np.random.RandomState(11)
def ev(Zm):
    p=lr.predict_proba(Zm); return dict(auc_hic=round(roc_auc_score(yt==2,p[:,2],sample_weight=wt),3),auc_geri=round(roc_auc_score(yt==1,p[:,1],sample_weight=wt),3),logloss=round(log_loss(yt,p,sample_weight=wt),3),tahmin_hic=round(float(np.average(p[:,2],weights=wt)*100),1),tahmin_geri=round(float(np.average(p[:,1],weights=wt)*100),1))
R=[]
for f in (0.1,0.3,0.5):
    mask={v:rs.rand(len(Xt))<f for v in VARS}; Xm=Xt.copy()
    for v in VARS: Xm.loc[mask[v],v]=np.nan
    Za=design(Xm); Zb=design(Xt).copy()
    for v in VARS:
        cs=[c for c in COLS if c.startswith(v+'=')]; Zb.loc[mask[v],cs]=[MU[c] for c in cs]
    R.append(dict(oran=int(f*100),kategori=ev(Za),ortalama=ev(Zb))); print(R[-1])
# bir soru hiç sorulmazsa (nüfus ortalamasıyla doldurularak)
LABV={v:M['vars'][v]['label'] for v in VARS}; drop=[]
for v in VARS:
    Zb=design(Xt).copy(); cs=[c for c in COLS if c.startswith(v+'=')]; Zb[cs]=[MU[c] for c in cs]; drop.append(dict(soru=LABV[v],**ev(Zb)))
drop.sort(key=lambda r:r['auc_hic']+r['auc_geri'])
S=json.load(open('out/stres.json')); S['soru_yok']=drop; S['eksik_karsilastirma']=R; S['gozlenen']=[round(float(np.average(yt==k,weights=wt)*100),1) for k in range(3)]; json.dump(S,open('out/stres.json','w'),ensure_ascii=False)
# tam veriyle kurulan nihai modelin sütun ortalamaları (araç için)
# araçtaki hesapla birebir karşılaştırma için test vakaları: bilinmeyen alanlar ortalama ile doldurulur
coef=np.array(M['coef']); ic=np.array(M['intercept']); T=[]
for t in J['tests']:
    o=t['o']; x=np.zeros(len(COLS))
    for j,c in enumerate(COLS):
        if c=='yas': x[j]=o['yas']-21
        elif c.startswith('yas_k'): x[j]=max(0,o['yas']-int(c[5:]))
        elif c=='cocuk': x[j]=o['cocuk']
        else:
            v,l=c.split('=')
            x[j]=MU[c] if o[v] is None else float(l!='NA' and int(l)==o[v])
    z=ic+coef@x; p=np.exp(z-z.max()); T.append(dict(o=o,p=np.round(p/p.sum(),4).tolist()))
o=dict(T[0]['o']); 
for v in ('gelir8','hiv_testi','checkup'): o[v]=None
x=np.array([ (o['yas']-21 if c=='yas' else max(0,o['yas']-int(c[5:])) if c.startswith('yas_k') else o['cocuk'] if c=='cocuk' else (MU[c] if o[c.split('=')[0]] is None else float(c.split('=')[1]!='NA' and int(c.split('=')[1])==o[c.split('=')[0]]))) for c in COLS]); z=ic+coef@x; p=np.exp(z-z.max()); T.append(dict(o=o,p=np.round(p/p.sum(),4).tolist()))
J['tests']=T
J['model']['colmeans']={c:round(float(m),5) for c,m in MU.items()}; json.dump(J,open('out/webmodel.json','w'),ensure_ascii=False); print('ok')
