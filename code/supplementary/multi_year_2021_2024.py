# Pooled 2021–2024 analysis with class derivation and temporal evaluation on 2024.
import pandas as pd, numpy as np, json, time, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
A=pd.read_pickle('out/cokyil_raw.pkl')
A['w']=A._LLCPWT/A.groupby('YIL')._LLCPWT.transform('mean')
def features(c):
    X=pd.DataFrame(index=c.index)
    def cat(v,miss=(7,9)):
        s=c[v].copy(); s[s.isin(miss)]=np.nan; return s
    X['yas']=c._AGE80; X['erkek']=(c._SEX==1).astype(float)
    X['irk']=cat('_RACE',(9,)).replace({6:np.nan})   # 'diğer' kategorisi 2022'de yok -> eksik
    X['egitim']=cat('EDUCA',(9,)); X['gelir']=cat('INCOME3',(77,99)); X['medeni']=cat('MARITAL',(9,)); X['istihdam']=cat('EMPLOY1',(9,))
    ch=c.CHILDREN.copy(); ch[ch==88]=0; ch[ch==99]=np.nan; X['cocuk']=ch.clip(upper=5)
    X['ev_sahibi']=cat('RENTHOM1'); X['kirsal']=cat('_URBSTAT')-1; X['dil_ispanyolca']=(c.QSTLANG==2).astype(float); X['gazi']=cat('VETERAN3'); X['eyalet']=c._STATE
    X['sigorta_turu']=cat('PRIMINS2',(77,99)); X['doktor']=cat('PERSDOC3'); X['maliyet_engeli']=cat('MEDCOST1'); X['checkup']=cat('CHECKUP1'); X['genel_saglik']=cat('GENHLTH')
    for v,n in [('PHYSHLTH','fiziksel_kotu_gun'),('MENTHLTH','ruhsal_kotu_gun')]:
        s=c[v].copy(); s[s==88]=0; s[s.isin([77,99])]=np.nan; X[n]=s
    X['bmi']=c._BMI5/100; X['sigara']=cat('_SMOKER3',(9,)); X['asiri_icme']=cat('_RFBING6',(9,)); X['egzersiz']=cat('EXERANY2')
    X['grip_asisi']=cat('FLUSHOT7'); X['hiv_testi']=cat('HIVTST7')
    for v,n in [('ADDEPEV3','depresyon'),('DIABETE4','diyabet'),('HAVARTH4','artrit'),('ASTHMA3','astim'),('CHCCOPD3','koah'),('_MICHD','kalp_hast'),
                ('DIFFWALK','yurume_guclugu'),('DECIDE','bilissel_gucluk'),('DIFFALON','yalniz_is_guclugu'),('DEAF','isitme'),('BLIND','gorme')]: X[n]=cat(v)
    return X
NUM=['yas','cocuk','fiziksel_kotu_gun','ruhsal_kotu_gun','bmi']
def serviks(d):
    b=d[(d._SEX==2)&(d._AGE80>=21)&(d._AGE80<=65)&(d.HADHYST2==2)]; y=pd.Series(np.nan,index=b.index); y[b.CERVSCRN==2]=2
    ok=(b.CERVSCRN==1)&b.CRVCLCNC.isin([1,2,3,4,5]); cur=ok&((b.CRVCLCNC<=3)|((b._AGE80>=30)&(b.CRVCLCNC==4)&(b.CRVCLHPV==1))); y[ok]=1; y[cur]=0; return b,y
def meme(d):
    b=d[(d._SEX==2)&(d._AGE80>=40)&(d._AGE80<=74)]; y=pd.Series(np.nan,index=b.index)
    y[b.HADMAM==2]=2; y[(b.HADMAM==1)&b.HOWLONG.isin([3,4,5])]=1; y[(b.HADMAM==1)&b.HOWLONG.isin([1,2])]=0; return b,y
def kolorektal(d):
    c=d[(d._AGE80>=45)&(d._AGE80<=75)]; I=lambda s,v:s.isin(v); T=[1,2,3,4,5]
    hc=(c.HADSIGM4==1)&I(c.COLNSIGM,[1,3]); hs=(c.HADSIGM4==1)&I(c.COLNSIGM,[2,3]); o=c.COLNCNCR==1
    hf=o&(c.SMALSTOL==1); hd=o&(c.STOOLDN2==1); hv=o&(c.VIRCOLO1==1)
    fit1=hf&(c.STOLTEST==1); sig10=hs&I(c.SIGMTES1,[1,2,3,4])
    met=(hc&I(c.COLNTES1,[1,2,3,4]))|(hs&I(c.SIGMTES1,[1,2,3]))|fit1|(hd&I(c.SDNATES1,[1,2,3]))|(hv&I(c.VCLNTES2,[1,2,3]))|(sig10&fit1)
    never=(c.HADSIGM4==2)&((c.COLNCNCR==2)|(o&(c.SMALSTOL==2)&(c.STOOLDN2==2)&(c.VIRCOLO1==2)))
    ever=hc|hs|hf|hd|hv
    unk=(hc&~I(c.COLNTES1,T))|(hs&~I(c.SIGMTES1,T))|(hf&~I(c.STOLTEST,T))|(hd&~I(c.SDNATES1,T))|(hv&~I(c.VCLNTES2,T))|((c.HADSIGM4==1)&~I(c.COLNSIGM,[1,2,3]))
    known=I(c.HADSIGM4,[1,2])&I(c.COLNCNCR,[1,2])
    y=pd.Series(np.nan,index=c.index); y[known&ever&~unk]=1; y[known&never]=2; y[met]=0; return c,y
COH={'serviks':serviks,'meme':meme,'kolorektal':kolorektal}
def met_(p,yt,wt):
    r={'logloss':log_loss(yt,p,sample_weight=wt,labels=[0,1,2])}
    for k,l in enumerate(['guncel','geri','hic']): r['AUC_'+l]=roc_auc_score(yt==k,p[:,k],sample_weight=wt)
    r['AUC_macro']=np.mean([r['AUC_guncel'],r['AUC_geri'],r['AUC_hic']]); return r
def fit_all(X,y,w,tr,te,tag,name,RES):
    feats=list(X.columns); catf=[f for f in feats if f not in NUM]
    Z=ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),NUM),
        ('c',Pipeline([('i',SimpleImputer(strategy='constant',fill_value=-1)),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=50,sparse_output=False))]),catf)]).fit(X.iloc[tr])
    Ztr=Z.transform(X.iloc[tr]).astype('float32'); Zte=Z.transform(X.iloc[te]).astype('float32')
    M={'Lojistik regresyon':LogisticRegression(C=0.5,max_iter=400),
       'XGBoost':xgb.XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=5,subsample=0.8,colsample_bytree=0.7,reg_lambda=5,n_jobs=2,tree_method='hist',random_state=0)}
    for mn,m in M.items():
        m.fit(Ztr,y[tr],sample_weight=w[tr]); r=met_(m.predict_proba(Zte),y[te],w[te]); r.update(kanser=name,senaryo=tag,model=mn); RES.append(r); print(name,tag,mn,round(r['AUC_macro'],3),flush=True)
    mlp=MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-2,early_stopping=True,n_iter_no_change=5,max_iter=60,random_state=0).fit(Ztr,y[tr])
    r=met_(mlp.predict_proba(Zte),y[te],w[te]); r.update(kanser=name,senaryo=tag,model='Yapay sinir ağı (MLP)'); RES.append(r); print(name,tag,'MLP',round(r['AUC_macro'],3),flush=True)
    XL=X.copy()
    for f in catf: XL[f]=XL[f].astype('category')
    tr2,va=train_test_split(tr,test_size=0.15,stratify=y[tr],random_state=1)
    par=dict(objective='multiclass',num_class=3,learning_rate=0.04,num_leaves=31,min_child_samples=80,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5,cat_smooth=30,verbose=-1,n_jobs=2,seed=42)
    g=lgb.train(par,lgb.Dataset(XL.iloc[tr2],y[tr2],weight=w[tr2]),2000,valid_sets=[lgb.Dataset(XL.iloc[va],y[va],weight=w[va])],callbacks=[lgb.early_stopping(60,verbose=False)])
    r=met_(g.predict(XL.iloc[te],num_iteration=g.best_iteration),y[te],w[te]); r.update(kanser=name,senaryo=tag,model='LightGBM'); RES.append(r); print(name,tag,'LightGBM',round(r['AUC_macro'],3),flush=True)
RES=[];SUM=[]
for name,fn in COH.items():
    b,y=fn(A); k=y.notna(); b=b[k]; y=y[k].astype(int).values; yil=b.YIL.values; w=b.w.values
    X=features(b); X=X[[f for f in X.columns if not(name!='kolorektal' and f=='erkek')]]
    for yr in sorted(set(yil)):
        m=yil==yr; SUM.append(dict(kanser=name,yil=int(yr),n=int(m.sum()),eyalet=int(b._STATE[m].nunique()),**{l:round(float(np.average(y[m]==j,weights=w[m])*100),1) for j,l in enumerate(['guncel','geri','hic'])}))
    print(pd.DataFrame(SUM).tail(4).to_string(),flush=True)
    idx=np.arange(len(y))
    fit_all(X,y,w,idx[yil<2024],idx[yil==2024],'Zamansal: 2021–2023 eğitim → 2024 test',name,RES)
    tr,te=train_test_split(idx,test_size=0.2,stratify=y,random_state=42)
    fit_all(X,y,w,tr,te,'Birleşik 2021–2024, rastgele %80/%20',name,RES)
    pd.DataFrame(RES).to_csv('out/cokyil_performans.csv',index=False); pd.DataFrame(SUM).to_csv('out/cokyil_ozet.csv',index=False)
print('BITTI')
