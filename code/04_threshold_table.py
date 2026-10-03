# Threshold table for the held-out validation set: flagged share, sensitivity, specificity and positive predictive value.
import pandas as pd, numpy as np, json, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'out'); sys.path.insert(0,'code')
from harmon import feats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
b=pd.read_pickle('out/serviks2020.pkl').reset_index(drop=True); y=b.y.astype(int).values; w=(b._LLCPWT/b._LLCPWT.mean()).values
X=feats(b,2020); M=json.load(open('out/webmodel.json'))['model']
def design(X):
    D={}
    for c in M['cols']:
        if c=='yas': D[c]=X.yas-21
        elif c.startswith('yas_k'): D[c]=np.clip(X.yas-int(c[5:]),0,None)
        elif c=='cocuk': D[c]=X.cocuk.fillna(0)
        else:
            v,l=c.split('='); lv=[int(q.split('=')[1]) for q in M['cols'] if q.startswith(v+'=') and not q.endswith('NA')]
            D[c]=((X[v].isna()|~X[v].isin(lv)) if l=='NA' else (X[v]==int(l))).astype(float)
    return pd.DataFrame(D)
Z=design(X); tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
p=LogisticRegression(C=1.0,max_iter=3000).fit(Z.iloc[tr],y[tr],sample_weight=w[tr]).predict_proba(Z.iloc[te]); yt=y[te]; wt=w[te]
out={}
for k,name in [(2,'never'),(1,'overdue')]:
    rows=[]; pos=(yt==k)
    for t in range(1,71):
        f=p[:,k]>=t/100; tp=wt[f&pos].sum(); fl=wt[f].sum()
        rows.append([t,round(fl/wt.sum()*100,1),round(tp/wt[pos].sum()*100,1),round(wt[~f&~pos].sum()/wt[~pos].sum()*100,1),round(tp/fl*100,1) if fl>0 else None])
    d=min(rows,key=lambda r:abs(r[1]-20))[0]; out[name]=dict(rows=rows,default=d); print(name,'varsayılan eşik',d,[r for r in rows if r[0] in (5,10,d,30)])
W=json.load(open('out/webdata.json')); W['thr']=out; json.dump(W,open('out/webdata.json','w'),ensure_ascii=False)
