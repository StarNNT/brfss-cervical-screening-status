# Ek soru bloklarının katkısı: temel (19 girdi) vs temel + blok, her blok kendi alt örnekleminde, 5 katlı ÇD (LightGBM)
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'out'); sys.path.insert(0,'code')
from harmon import feats
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
# Reuse the single audited label definition from script 01; do not reimplement it here.
b=pd.read_pickle('out/serviks2020.pkl').reset_index(drop=True); y=b.y.astype(int).values; w=(b._LLCPWT/b._LLCPWT.mean()).values; print(len(y))
TOOL=['yas','cocuk','irk','egitim','gelir8','medeni','istihdam','ev_sahibi','kirsal','dil_ispanyolca','sigortasiz','doktor','maliyet_engeli','checkup','genel_saglik','sigara','grip_asisi','hiv_testi','dis_hekimi']
XB=feats(b,2020)[TOOL]
def num(v,miss=(7,9),zero=None):
    s=b[v].astype(float).copy()
    if zero is not None: s[s==zero]=0
    s[s.isin(miss)]=np.nan; return s
E=pd.DataFrame(index=b.index)
for v in ['CVDINFR4','CVDCRHD4','CVDSTRK3','ASTHMA3','CHCSCNCR','CHCOCNCR','CHCCOPD2','HAVARTH4','ADDEPEV3','CHCKDNY2','DIABETE4','DEAF','BLIND','DECIDE','DIFFWALK','DIFFDRES','DIFFALON']: E[v]=num(v)
E['PHYSHLTH']=num('PHYSHLTH',(77,99),88); E['MENTHLTH']=num('MENTHLTH',(77,99),88)
E['EXERANY2']=num('EXERANY2'); E['SLEPTIM1']=num('SLEPTIM1',(77,99)); E['_BMI5']=b._BMI5/100; E['_RFBING5']=num('_RFBING5',(9,)); E['_RFDRHV7']=num('_RFDRHV7',(9,)); E['SEATBELT']=num('SEATBELT',(7,9))
E['HIVRISK5']=num('HIVRISK5'); E['PREGNANT']=num('PREGNANT')
E['HADMAM']=num('HADMAM'); E['HOWLONG']=num('HOWLONG').where(b.HADMAM==1,6)   # 6 = hiç mamografi yok
E.loc[b.HADMAM.isna()|b.HADMAM.isin([7,9]),'HOWLONG']=np.nan
E['HPVADVC4']=num('HPVADVC4',(7,9)); E['HPVADSHT']=num('HPVADSHT',(77,99))
E['SOFEMALE']=num('SOFEMALE'); E['TRNSGNDR']=num('TRNSGNDR')
ace=['ACEDEPRS','ACEDRINK','ACEDRUGS','ACEPRISN','ACEDIVRC','ACEPUNCH','ACEHURT1','ACESWEAR','ACETOUCH','ACETTHEM','ACEHVSEX']
for v in ace: E[v]=num(v,(7,9) if v!='ACEDIVRC' else (7,8,9))
pos=pd.concat([(E[v]==1) if v in ace[:5] else (E[v]>=2) for v in ace],axis=1); E['ACE_SKOR']=pos.sum(axis=1).where(E[ace].notna().any(axis=1))
E['HLTHCVR1']=num('HLTHCVR1',(77,99)); E['HANE_YETISKIN']=b.NUMADULT.fillna(b.HHADULT).where(lambda s:s<77).clip(upper=6)
E['ECIGNOW']=num('ECIGNOW'); E['MARIJAN1']=num('MARIJAN1',(77,99),88); E['TETANUS1']=num('TETANUS1'); E['CAREGIV1']=num('CAREGIV1',(7,8,9)); E['eyalet']=b._STATE
CATS=['irk','medeni','istihdam','eyalet','HLTHCVR1','SOFEMALE']
B={'Eyalet':(['eyalet'],None),
 'Kronik hastalıklar ve engellilik (19 soru)':(['CVDINFR4','CVDCRHD4','CVDSTRK3','ASTHMA3','CHCSCNCR','CHCOCNCR','CHCCOPD2','HAVARTH4','ADDEPEV3','CHCKDNY2','DIABETE4','DEAF','BLIND','DECIDE','DIFFWALK','DIFFDRES','DIFFALON','PHYSHLTH','MENTHLTH'],None),
 'Yaşam tarzı: egzersiz, uyku, BMI, alkol, emniyet kemeri (6 soru)':(['EXERANY2','SLEPTIM1','_BMI5','_RFBING5','_RFDRHV7','SEATBELT'],None),
 'HIV risk davranışı ve gebelik (2 soru)':(['HIVRISK5','PREGNANT'],None),
 'Hane yapısı: hanedeki yetişkin sayısı':(['HANE_YETISKIN'],None),
 'Tüm çekirdek ek sorular birlikte (29 soru)':(None,None),
 'Mamografi öyküsü (40–65 yaş)':(['HADMAM','HOWLONG'],(b._AGE80>=40)&E.HADMAM.notna()),
 'HPV aşısı modülü (18–49 yaş, modülü soran eyaletler)':(['HPVADVC4','HPVADSHT'],E.HPVADVC4.notna()|(b.HPVADVC4.isin([7,9]))),
 'Cinsel yönelim ve cinsiyet kimliği modülü':(['SOFEMALE','TRNSGNDR'],b.SOFEMALE.notna()),
 'Çocukluk çağı olumsuz deneyimleri modülü (11 soru + skor)':(ace+['ACE_SKOR'],b[ace].notna().any(axis=1)),
 'Sigorta türü modülü':(['HLTHCVR1'],b.HLTHCVR1.notna()),
 'E-sigara, esrar, tetanos aşısı, bakım verme (modüller)':(['ECIGNOW','MARIJAN1','TETANUS1','CAREGIV1'],b[['ECIGNOW','MARIJAN1','TETANUS1','CAREGIV1']].notna().any(axis=1))}
core=[c for k,(c,m) in B.items() if c and m is None for c in c]; B['Tüm çekirdek ek sorular birlikte (29 soru)']=(core,None)
par=dict(objective='multiclass',num_class=3,learning_rate=0.05,num_leaves=31,min_child_samples=60,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5,cat_smooth=30,verbose=-1,n_jobs=2,seed=42)
def prep(X):
    X=X.copy()
    for c in X.columns:
        if c in CATS: X[c]=X[c].astype('category')
    return X
def cv(X,yy,ww):
    X=prep(X); R=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=7).split(X,yy):
        p=lgb.train(par,lgb.Dataset(X.iloc[tr],yy[tr],weight=ww[tr]),160).predict(X.iloc[te]); a=[roc_auc_score(yy[te]==k,p[:,k],sample_weight=ww[te]) for k in range(3)]
        R.append(a+[np.mean(a),log_loss(yy[te],p,sample_weight=ww[te],labels=[0,1,2])])
    return np.array(R)
OUT=[];cache={}
for name,(cols,mask) in B.items():
    m=np.ones(len(y),bool) if mask is None else mask.values
    key='all' if mask is None else name
    if key not in cache: cache[key]=cv(XB[m],y[m],w[m])
    r0=cache[key]; r1=cv(pd.concat([XB[m],E.loc[m,cols]],axis=1),y[m],w[m]); dlt=r1-r0
    o=dict(blok=name,n=int(m.sum()),eyalet=int(b._STATE[m].nunique()),hic_oran=round(float(np.average(y[m]==2,weights=w[m])*100),1),
           temel=[round(float(x),4) for x in r0.mean(0)],ek=[round(float(x),4) for x in r1.mean(0)],fark=[round(float(x),4) for x in dlt.mean(0)],fark_ss=[round(float(x),4) for x in dlt.std(0)])
    OUT.append(o); print(name,'| n',o['n'],'| temel makro',o['temel'][3],'→',o['ek'][3],'Δ',o['fark'][3],'±',o['fark_ss'][3],'| Δhiç',o['fark'][2],'Δgeri',o['fark'][1],'Δlogloss',o['fark'][4],flush=True)
    json.dump(OUT,open('out/ek_bloklar.json','w'),ensure_ascii=False)
print('BITTI')
