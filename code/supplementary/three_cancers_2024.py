# Üç kanser taraması (serviks, meme, kolorektal) için ortak özellik seti ve geniş model karşılaştırması
import pandas as pd, numpy as np, json, time, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
df=pd.read_pickle('out/tarama_raw.pkl')
def features(c):
    X=pd.DataFrame(index=c.index)
    def cat(v,miss=(7,9)):
        s=c[v].copy(); s[s.isin(miss)]=np.nan; return s
    X['yas']=c._AGE80; X['erkek']=(c._SEX==1).astype(float)
    X['irk']=cat('_RACE',(9,)); X['egitim']=cat('EDUCA',(9,)); X['gelir']=cat('INCOME3',(77,99)); X['medeni']=cat('MARITAL',(9,)); X['istihdam']=cat('EMPLOY1',(9,))
    ch=c.CHILDREN.copy(); ch[ch==88]=0; ch[ch==99]=np.nan; X['cocuk']=ch.clip(upper=5)
    X['ev_sahibi']=cat('RENTHOM1'); X['kirsal']=cat('_URBSTAT')-1; X['dil_ispanyolca']=(c.QSTLANG==2).astype(float); X['gazi']=cat('VETERAN3'); X['eyalet']=c._STATE
    X['sigorta_turu']=cat('PRIMINS2',(77,99)); X['doktor']=cat('PERSDOC3'); X['maliyet_engeli']=cat('MEDCOST1'); X['checkup']=cat('CHECKUP1'); X['genel_saglik']=cat('GENHLTH')
    for v,n in [('PHYSHLTH','fiziksel_kotu_gun'),('MENTHLTH','ruhsal_kotu_gun')]:
        s=c[v].copy(); s[s==88]=0; s[s.isin([77,99])]=np.nan; X[n]=s
    X['bmi']=c._BMI5/100; X['sigara']=cat('_SMOKER3',(9,)); X['asiri_icme']=cat('_RFBING6',(9,)); X['egzersiz']=cat('EXERANY2')
    X['grip_asisi']=cat('FLUSHOT7'); X['hiv_testi']=cat('HIVTST7'); X['dis_hekimi']=cat('LASTDEN4',(7,9))
    for v,n in [('ADDEPEV3','depresyon'),('DIABETE4','diyabet'),('HAVARTH4','artrit'),('ASTHMA3','astim'),('CHCCOPD3','koah'),('CHCOCNC1','kanser_oykusu'),('_MICHD','kalp_hast'),
                ('DIFFWALK','yurume_guclugu'),('DECIDE','bilissel_gucluk'),('DIFFALON','yalniz_is_guclugu'),('DEAF','isitme'),('BLIND','gorme')]: X[n]=cat(v)
    return X
NUM=['yas','cocuk','fiziksel_kotu_gun','ruhsal_kotu_gun','bmi']
# ---- kohortlar: y 0=güncel 1=geri kalmış 2=hiç
def serviks(d):
    b=d[(d._SEX==2)&(d._AGE80>=21)&(d._AGE80<=65)&(d.HADHYST2==2)].copy()
    y=pd.Series(np.nan,index=b.index); y[b.CERVSCRN==2]=2
    ok=(b.CERVSCRN==1)&b.CRVCLCNC.isin([1,2,3,4,5]); cur=ok&((b.CRVCLCNC<=3)|((b._AGE80>=30)&(b.CRVCLCNC==4)&(b.CRVCLHPV==1)))
    y[ok]=1; y[cur]=0; return b,y
def meme(d):
    b=d[(d._SEX==2)&(d._AGE80>=40)&(d._AGE80<=74)].copy(); y=pd.Series(np.nan,index=b.index)
    y[b.HADMAM==2]=2; y[(b.HADMAM==1)&b.HOWLONG.isin([3,4,5])]=1; y[(b.HADMAM==1)&b.HOWLONG.isin([1,2])]=0; return b,y
def kolorektal(d):
    b=d[(d._AGE80>=45)&(d._AGE80<=75)].copy(); y=b._CRCREC3.map({1:0,2:1,3:2}); return b,y
COH={'serviks':serviks,'meme':meme,'kolorektal':kolorektal}
def met(p,yt,wt):
    r={'logloss':log_loss(yt,p,sample_weight=wt,labels=[0,1,2])}
    for k,l in enumerate(['guncel','geri','hic']):
        r['AUC_'+l]=roc_auc_score(yt==k,p[:,k],sample_weight=wt); r['PRAUC_'+l]=average_precision_score(yt==k,p[:,k],sample_weight=wt); r['Brier_'+l]=np.average(((yt==k)-p[:,k])**2,weights=wt)
    r['AUC_macro']=np.mean([r['AUC_guncel'],r['AUC_geri'],r['AUC_hic']]); return r
ALL=[];SUM=[];IMP={}
for name,fn in COH.items():
    b,y=fn(df); k=y.notna(); b=b[k]; y=y[k].astype(int).values; X=features(b); w=(b._LLCPWT/b._LLCPWT.mean()).values
    feats=[f for f in X.columns if not(name!='kolorektal' and f=='erkek')]; X=X[feats]; catf=[f for f in feats if f not in NUM]
    dist=[float(np.average(y==j,weights=w)*100) for j in range(3)]
    SUM.append(dict(kanser=name,n=len(y),n_guncel=int((y==0).sum()),n_geri=int((y==1).sum()),n_hic=int((y==2).sum()),guncel=dist[0],geri=dist[1],hic=dist[2])); print(SUM[-1],flush=True)
    tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
    pre=lambda scale: ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median'))]+([('s',StandardScaler())] if scale else [])),NUM),
        ('c',Pipeline([('i',SimpleImputer(strategy='constant',fill_value=-1)),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=50,sparse_output=False))]),catf)])
    Z=pre(True).fit(X.iloc[tr]); Ztr=Z.transform(X.iloc[tr]); Zte=Z.transform(X.iloc[te])
    models={'Lojistik regresyon':LogisticRegression(C=0.5,max_iter=500),
            'Karar ağacı':DecisionTreeClassifier(max_depth=8,min_samples_leaf=200,random_state=0),
            'Naive Bayes':GaussianNB(),
            'Random forest':RandomForestClassifier(n_estimators=300,min_samples_leaf=20,max_features='sqrt',n_jobs=2,random_state=0),
            'XGBoost':xgb.XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=5,subsample=0.8,colsample_bytree=0.7,reg_lambda=5,n_jobs=2,tree_method='hist',random_state=0)}
    for mn,m in models.items():
        t=time.time(); m.fit(Ztr,y[tr],sample_weight=w[tr]); p=m.predict_proba(Zte); r=met(p,y[te],w[te]); r.update(kanser=name,model=mn,sure_sn=round(time.time()-t)); ALL.append(r); print(name,mn,round(r['AUC_macro'],3),r['sure_sn'],flush=True)
    # MLP (sample_weight desteklemez; ağırlıksız eğitilir, ağırlıklı değerlendirilir)
    t=time.time(); mlp=MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-2,early_stopping=True,n_iter_no_change=5,max_iter=80,random_state=0).fit(Ztr,y[tr]); r=met(mlp.predict_proba(Zte),y[te],w[te]); r.update(kanser=name,model='Yapay sinir ağı (MLP)',sure_sn=round(time.time()-t)); ALL.append(r); print(name,'MLP',round(r['AUC_macro'],3),flush=True)
    # LightGBM (yerel kategorik)
    t=time.time(); XL=X.copy()
    for f in catf: XL[f]=XL[f].astype('category')
    tr2,va=train_test_split(tr,test_size=0.15,stratify=y[tr],random_state=1)
    par=dict(objective='multiclass',num_class=3,learning_rate=0.04,num_leaves=31,min_child_samples=80,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5,cat_smooth=30,verbose=-1,n_jobs=2,seed=42)
    g=lgb.train(par,lgb.Dataset(XL.iloc[tr2],y[tr2],weight=w[tr2]),2000,valid_sets=[lgb.Dataset(XL.iloc[va],y[va],weight=w[va])],callbacks=[lgb.early_stopping(60,verbose=False)])
    r=met(g.predict(XL.iloc[te],num_iteration=g.best_iteration),y[te],w[te]); r.update(kanser=name,model='LightGBM',sure_sn=round(time.time()-t)); ALL.append(r); print(name,'LightGBM',round(r['AUC_macro'],3),flush=True)
    gi=pd.Series(g.feature_importance('gain'),index=feats); IMP[name]=(gi/gi.sum()*100).round(2).sort_values(ascending=False).to_dict()
    pd.DataFrame(ALL).to_csv('out/genis_performans.csv',index=False); pd.DataFrame(SUM).to_csv('out/genis_ozet.csv',index=False); json.dump(IMP,open('out/genis_onem.json','w'))
print('BITTI')
