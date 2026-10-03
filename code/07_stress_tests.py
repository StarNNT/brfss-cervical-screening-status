# Stres testleri: nihai lojistik model, 2020 iç doğrulama kümesi. Gerçek veri kontrollü biçimde bozulur; yeni soru uydurulmaz.
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'out'); sys.path.insert(0,'code')
from harmon import feats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
b=pd.read_pickle('out/serviks2020.pkl').reset_index(drop=True); y=b.y.astype(int).values; w=(b._LLCPWT/b._LLCPWT.mean()).values
X=feats(b,2020); M=json.load(open('out/webmodel.json'))['model']; COLS=M['cols']
VARS=[v for v in M['vars']]; LEV={v:[c for c,_ in M['vars'][v]['levels']] for v in VARS}
def design(X):
    D={}
    for c in COLS:
        if c=='yas': D[c]=X.yas-21
        elif c.startswith('yas_k'): D[c]=np.clip(X.yas-int(c[5:]),0,None)
        elif c=='cocuk': D[c]=X.cocuk.fillna(0)
        else:
            v,l=c.split('='); D[c]=((X[v].isna()|~X[v].isin(LEV[v])) if l=='NA' else (X[v]==int(l))).astype(float)
    return pd.DataFrame(D)
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
Z=design(X); lr=LogisticRegression(C=1.0,max_iter=3000).fit(Z.iloc[tr],y[tr],sample_weight=w[tr])
Xt=X.iloc[te].reset_index(drop=True); yt=y[te]; wt=w[te]; rs=np.random.RandomState(11)
def ev(Xm,ww=None):
    ww=wt if ww is None else ww; p=lr.predict_proba(design(Xm))
    return dict(auc_hic=round(roc_auc_score(yt==2,p[:,2],sample_weight=ww),3),auc_geri=round(roc_auc_score(yt==1,p[:,1],sample_weight=ww),3),logloss=round(log_loss(yt,p,sample_weight=ww),3),
                tahmin_hic=round(float(np.average(p[:,2],weights=ww)*100),1),gozlenen_hic=round(float(np.average(yt==2,weights=ww)*100),1),tahmin_geri=round(float(np.average(p[:,1],weights=ww)*100),1),gozlenen_geri=round(float(np.average(yt==1,weights=ww)*100),1))
base=ev(Xt); print('temel',base); OUT=dict(base=base,n=int(len(yt)))
LAB={v:M['vars'][v]['label'] for v in VARS}
# A1. tek bir soru hiç sorulmazsa
A=[]
for v in VARS:
    Xm=Xt.copy(); Xm[v]=np.nan; r=ev(Xm); A.append(dict(soru=LAB[v],**r))
A.sort(key=lambda r:r['auc_hic']+r['auc_geri']); OUT['soru_yok']=A
# A2. rastgele eksik yanıt
A2=[]
for f in (0.1,0.3,0.5):
    Xm=Xt.copy()
    for v in VARS: Xm.loc[rs.rand(len(Xm))<f,v]=np.nan
    A2.append(dict(senaryo=f'Her sorunun %{int(f*100)} kadarı yanıtsız',**ev(Xm)))
OUT['eksik']=A2
# B. hatalı yanıt
def shift(s,lv,frac,step=1):
    s=s.copy(); m=(rs.rand(len(s))<frac)&s.notna(); idx=s[m].map({c:i for i,c in enumerate(lv)}); idx=(idx+rs.choice([-step,step],m.sum())).clip(0,len(lv)-1); s[m]=idx.map(dict(enumerate(lv))).values; return s
Bk=[]
Xm=Xt.copy(); Xm['checkup']=shift(Xm.checkup,[1,2,3,4,8],0.2); Bk.append(dict(senaryo='Son check-up zamanı: kadınların %20’sinde bir basamak kayık',**ev(Xm)))
Xm=Xt.copy(); Xm['gelir8']=shift(Xm.gelir8,list(range(1,9)),0.3); Bk.append(dict(senaryo='Gelir: kadınların %30’unda bir basamak kayık',**ev(Xm)))
Xm=Xt.copy(); m=rs.rand(len(Xm))<0.1; Xm.loc[m&Xm.hiv_testi.notna(),'hiv_testi']=3-Xm.loc[m&Xm.hiv_testi.notna(),'hiv_testi']; Bk.append(dict(senaryo='HIV testi: kadınların %10’unda yanıt ters',**ev(Xm)))
Xm=Xt.copy(); Xm['yas']=(Xm.yas+rs.randint(-3,4,len(Xm))).clip(21,65); Bk.append(dict(senaryo='Yaş: ±3 yıl rastgele hata',**ev(Xm)))
Xm=Xt.copy(); Xm['checkup']=shift(Xm.checkup,[1,2,3,4,8],0.2); Xm['gelir8']=shift(Xm.gelir8,list(range(1,9)),0.3); Xm['dis_hekimi']=shift(Xm.dis_hekimi,[1,2,3,4,8],0.2); m=rs.rand(len(Xm))<0.1; Xm.loc[m&Xm.hiv_testi.notna(),'hiv_testi']=3-Xm.loc[m&Xm.hiv_testi.notna(),'hiv_testi']; Xm['yas']=(Xm.yas+rs.randint(-3,4,len(Xm))).clip(21,65)
Bk.append(dict(senaryo='Hepsi birlikte (check-up, diş hekimi, gelir, HIV testi, yaş)',**ev(Xm))); OUT['hatali']=Bk
# C. nüfus kayması (yeniden ağırlıklandırma)
S={'Daha genç nüfus (30 yaş altı 3 kat ağırlık)':Xt.yas<30,'Daha çok sigortasız (3 kat)':Xt.sigortasiz==1,'Daha çok Hispanik (3 kat)':Xt.irk==8,'Daha düşük gelirli (25 bin $ altı 3 kat)':Xt.gelir8<=4,'Daha kırsal (3 kat)':Xt.kirsal==1,'Düzenli doktoru olmayanlar (3 kat)':Xt.doktor==3}
OUT['kayma']=[dict(senaryo=k,pay=round(float(np.average(m.values,weights=wt*np.where(m.values,3,1))*100),1),**ev(Xt,wt*np.where(m.values,3,1))) for k,m in S.items()]
for k in ('eksik','hatali','kayma'): print(pd.DataFrame(OUT[k]).to_string())
print(pd.DataFrame(A).head(8).to_string())
# D. kısa form: ileri doğru seçim (eğitim kümesinin içinden ayrılan doğrulama parçasında log-loss)
grp={}
for c in COLS: grp.setdefault('yas' if c.startswith('yas') else c.split('=')[0],[]).append(c)
LAB2=dict(LAB,yas='Yaş',cocuk='Hanedeki çocuk sayısı')
sub=rs.choice(tr,30000,replace=False); s_tr,s_va=sub[:22000],sub[22000:]
chosen=[];rest=list(grp); path=[]
while len(chosen)<9:
    best=None
    for g in rest:
        cols=[c for q in chosen+[g] for c in grp[q]]
        mm=LogisticRegression(C=1.0,max_iter=300).fit(Z.iloc[s_tr][cols],y[s_tr],sample_weight=w[s_tr]); ll=log_loss(y[s_va],mm.predict_proba(Z.iloc[s_va][cols]),sample_weight=w[s_va],labels=[0,1,2])
        if best is None or ll<best[0]: best=(ll,g)
    chosen.append(best[1]); rest.remove(best[1])
    cols=[c for q in chosen for c in grp[q]]; mm=LogisticRegression(C=1.0,max_iter=1500).fit(Z.iloc[tr][cols],y[tr],sample_weight=w[tr]); p=mm.predict_proba(Z.iloc[te][cols])
    a=[roc_auc_score(yt==k,p[:,k],sample_weight=wt) for k in range(3)]
    path.append(dict(k=len(chosen),eklenen=LAB2[best[1]],auc_hic=round(a[2],3),auc_geri=round(a[1],3),auc_guncel=round(a[0],3),auc_ort=round(float(np.mean(a)),3),logloss=round(log_loss(yt,p,sample_weight=wt),3))); print(path[-1],flush=True)
    OUT['kisa_form']=path; json.dump(OUT,open('out/stres.json','w'),ensure_ascii=False)
# 9. adımdan sonra kazanımlar çok küçük: kalan sorular, o noktadaki tek adımlık katkılarına göre sıralanıp eklenir
sc=[]
for g in rest:
    cols=[c for q in chosen+[g] for c in grp[q]]; mm=LogisticRegression(C=1.0,max_iter=200).fit(Z.iloc[s_tr][cols],y[s_tr],sample_weight=w[s_tr]); sc.append((log_loss(y[s_va],mm.predict_proba(Z.iloc[s_va][cols]),sample_weight=w[s_va],labels=[0,1,2]),g))
for _,g in sorted(sc):
    chosen.append(g); cols=[c for q in chosen for c in grp[q]]; mm=LogisticRegression(C=1.0,max_iter=800).fit(Z.iloc[tr][cols],y[tr],sample_weight=w[tr]); p=mm.predict_proba(Z.iloc[te][cols]); a=[roc_auc_score(yt==k,p[:,k],sample_weight=wt) for k in range(3)]
    path.append(dict(k=len(chosen),eklenen=LAB2[g],auc_hic=round(a[2],3),auc_geri=round(a[1],3),auc_guncel=round(a[0],3),auc_ort=round(float(np.mean(a)),3),logloss=round(log_loss(yt,p,sample_weight=wt),3))); print(path[-1],flush=True)
json.dump(OUT,open('out/stres.json','w'),ensure_ascii=False)
print('BITTI')
