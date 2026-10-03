# Final model, held-out validation, 2024 temporal evaluation, profiles and web export.
import pandas as pd, numpy as np, sys, json, warnings, os; warnings.filterwarnings('ignore'); sys.path.insert(0,'code')
from harmon import feats, NUM
import lightgbm as lgb, shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
U=os.environ.get('BRFSS_DATA_DIR','.')
b=pd.read_pickle('out/serviks2020.pkl').reset_index(drop=True); y=b.y.astype(int).values; wr=b._LLCPWT.values; w=wr/wr.mean()
a=pd.read_pickle('out/analitik.pkl').reset_index(drop=True); ya=a.y.map({'current':0,'overdue':1,'never':2}).values; wa=(a._LLCPWT/a._LLCPWT.mean()).values
XF=feats(b,2020); XA=feats(a,2024)
V={'irk':('Irk / etnisite',{1:'Beyaz (Hispanik olmayan)',2:'Siyah (Hispanik olmayan)',8:'Hispanik',4:'Asyalı',3:'Amerikan Yerlisi / Alaska Yerlisi',5:'Havaili / Pasifik Adalı',7:'Çok ırklı',6:'Diğer'}),
'egitim':('Eğitim',{1:'Okula gitmemiş',2:'İlkokul–ortaokul',3:'Lise terk',4:'Lise mezunu',5:'Üniversite terk / ön lisans',6:'Üniversite mezunu'}),
'gelir8':('Yıllık hane geliri',{1:'< 10 bin $',2:'10–15 bin $',3:'15–20 bin $',4:'20–25 bin $',5:'25–35 bin $',6:'35–50 bin $',7:'50–75 bin $',8:'75 bin $ ve üzeri'}),
'medeni':('Medeni durum',{1:'Evli',2:'Boşanmış',3:'Dul',4:'Ayrı yaşıyor',5:'Hiç evlenmemiş',6:'Partneriyle yaşıyor'}),
'istihdam':('İstihdam',{1:'Ücretli çalışan',2:'Kendi işinde',3:'İşsiz (1 yıl+)',4:'İşsiz (<1 yıl)',5:'Ev hanımı',6:'Öğrenci',7:'Emekli',8:'Çalışamıyor'}),
'ev_sahibi':('Konut',{1:'Ev sahibi',2:'Kiracı',3:'Diğer'}),'kirsal':('Yerleşim',{0:'Kentsel',1:'Kırsal'}),'dil_ispanyolca':('Görüşme dili',{0:'İngilizce',1:'İspanyolca'}),
'sigortasiz':('Sağlık sigortası',{0:'Var',1:'Yok'}),'doktor':('Düzenli doktor',{1:'Bir doktoru var',2:'Birden fazla doktoru var',3:'Yok'}),
'maliyet_engeli':('Son 12 ayda maliyet yüzünden doktora gidememe',{1:'Evet',2:'Hayır'}),'checkup':('Son genel kontrol (check-up)',{1:'Son 1 yıl içinde',2:'1–2 yıl önce',3:'2–5 yıl önce',4:'5 yıldan eski',8:'Hiç'}),
'genel_saglik':('Genel sağlık algısı',{1:'Mükemmel',2:'Çok iyi',3:'İyi',4:'Orta',5:'Kötü'}),'sigara':('Sigara',{1:'Her gün içiyor',2:'Bazı günler içiyor',3:'Bırakmış',4:'Hiç içmemiş'}),
'grip_asisi':('Son 12 ayda grip aşısı',{1:'Evet',2:'Hayır'}),'hiv_testi':('Hiç HIV testi yaptırmış mı',{1:'Evet',2:'Hayır'}),'dis_hekimi':('Son diş hekimi ziyareti',{1:'Son 1 yıl içinde',2:'1–2 yıl önce',3:'2–5 yıl önce',4:'5 yıldan eski',8:'Hiç'})}
knots=[23,26,30,40,50,60]
def design(X):
    P=[pd.DataFrame({'yas':X.yas-21,**{f'yas_k{k}':np.clip(X.yas-k,0,None) for k in knots},'cocuk':X.cocuk.fillna(0)})]
    for v,(l,lv) in V.items():
        s=X[v]; P.append(pd.DataFrame({f'{v}={c}':(s==c).astype(float) for c in lv}|{f'{v}=NA':(s.isna()|~s.isin(list(lv))).astype(float)}))
    return pd.concat(P,axis=1)
Z=design(XF); ZA=design(XA)
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
lr_h=LogisticRegression(C=1.0,max_iter=3000).fit(Z.iloc[tr],y[tr],sample_weight=w[tr]); ph=lr_h.predict_proba(Z.iloc[te])
lr=LogisticRegression(C=1.0,max_iter=3000).fit(Z,y,sample_weight=w); pa=lr.predict_proba(ZA)
rs=np.random.RandomState(1)
def auc_ci(t,p,ww,B=200):
    v=roc_auc_score(t,p,sample_weight=ww); n=len(t); bs=[]
    for _ in range(B):
        i=rs.randint(0,n,n)
        if t[i].min()==t[i].max(): continue
        bs.append(roc_auc_score(t[i],p[i],sample_weight=ww[i]))
    return [round(v,3),round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)]
LAB=['guncel','geri','hic']
internal={l:auc_ci((y[te]==k).astype(int),ph[:,k],w[te]) for k,l in enumerate(LAB)}; internal['logloss']=round(log_loss(y[te],ph,sample_weight=w[te]),3)
ext={l:auc_ci((ya==k).astype(int),pa[:,k],wa) for k,l in enumerate(LAB)}; ext['logloss']=round(log_loss(ya,pa,sample_weight=wa),3)
ev=ya!=2; pov=pa[ev,1]/(pa[ev,0]+pa[ev,1]); ext['geri_taranmislar']=auc_ci((ya[ev]==1).astype(int),pov,wa[ev])
evh=y[te]!=2; internal['geri_taranmislar']=auc_ci((y[te][evh]==1).astype(int),ph[evh,1]/(ph[evh,0]+ph[evh,1]),w[te][evh])
def calib(t,p,ww,q=10):
    g=pd.DataFrame({'p':p,'o':t.astype(float),'w':ww}); g['q']=pd.qcut(g.p,q,duplicates='drop')
    return [[round(np.average(d.p,weights=d.w)*100,1),round(np.average(d.o,weights=d.w)*100,1)] for _,d in g.groupby('q',observed=True)]
cal=[calib(y[te]==k,ph[:,k],w[te]) for k in range(3)]
ext['cal_geri_taranmislar']=calib(ya[ev]==1,pov,wa[ev]); ext['ort_tahmin']=[round(float(x)*100,1) for x in np.average(pa,axis=0,weights=wa)]; ext['gozlenen']=[round(float(np.average(ya==k,weights=wa))*100,1) for k in range(3)]
print('iç',internal); print('dış',{k:v for k,v in ext.items() if k!='cal_geri_taranmislar'})
# Ranking performance (lift)
lift=[]
for k,l in [(2,'hic'),(1,'geri')]:
    o=np.argsort(-ph[:,k]); cw=np.cumsum(w[te][o])/w[te].sum(); t=(y[te][o]==k)*w[te][o]; base=t.sum()/w[te].sum()
    for f in (0.1,0.2,0.3):
        m=cw<=f; lift.append(dict(sinif=l,dilim=int(f*100),yakalama=round(float(t[m].sum()/t.sum()*100),1),ppv=round(float(t[m].sum()/w[te][o][m].sum()*100),1),taban=round(float(base*100),1)))
print(pd.DataFrame(lift).to_string())
# Subgroup performance
Xt=XF.iloc[te]; sub=[]
G={'Beyaz':Xt.irk==1,'Siyah':Xt.irk==2,'Hispanik':Xt.irk==8,'Asyalı':Xt.irk==4,'Sigortalı':Xt.sigortasiz==0,'Sigortasız':Xt.sigortasiz==1,'21–29 yaş':Xt.yas<30,'30–49 yaş':(Xt.yas>=30)&(Xt.yas<50),'50–65 yaş':Xt.yas>=50,'Kentsel':Xt.kirsal==0,'Kırsal':Xt.kirsal==1,'İngilizce görüşme':Xt.dil_ispanyolca==0,'İspanyolca görüşme':Xt.dil_ispanyolca==1}
for g,m in G.items():
    m=m.values; sub.append(dict(grup=g,n=int(m.sum()),hic_oran=round(float(np.average(y[te][m]==2,weights=w[te][m])*100),1),AUC_hic=round(roc_auc_score(y[te][m]==2,ph[m,2],sample_weight=w[te][m]),3),AUC_geri=round(roc_auc_score(y[te][m]==1,ph[m,1],sample_weight=w[te][m]),3)))
print(pd.DataFrame(sub).to_string())
# LightGBM and SHAP analysis using the extended predictor set
XL=XF.copy()
for f in XL.columns:
    if f not in NUM: XL[f]=XL[f].astype('category')
tr2,va=train_test_split(tr,test_size=0.15,stratify=y[tr],random_state=1)
par=dict(objective='multiclass',num_class=3,learning_rate=0.04,num_leaves=31,min_child_samples=80,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5,cat_smooth=30,verbose=-1,n_jobs=2,seed=42)
g=lgb.train(par,lgb.Dataset(XL.iloc[tr2],y[tr2],weight=w[tr2]),2000,valid_sets=[lgb.Dataset(XL.iloc[va],y[va],weight=w[va])],callbacks=[lgb.early_stopping(60,verbose=False)])
pg=g.predict(XL.iloc[te],num_iteration=g.best_iteration); lgbm={l:round(roc_auc_score(y[te]==k,pg[:,k],sample_weight=w[te]),3) for k,l in enumerate(LAB)}; lgbm['logloss']=round(log_loss(y[te],pg,sample_weight=w[te]),3); print('lgbm',lgbm)
idx=np.random.RandomState(0).choice(te,6000,replace=False); sv=np.array(shap.TreeExplainer(g).shap_values(XL.iloc[idx]))
if sv.shape[0]==3: sv=np.transpose(sv,(1,2,0))
imp=pd.DataFrame(np.abs(sv).mean(0),index=XL.columns); imp['t']=imp.sum(axis=1); imp=imp.sort_values('t',ascending=False)
shap_out=[[i]+[round(float(x),4) for x in r[:3]] for i,r in zip(imp.index,imp.values)]
# Design-based confidence intervals
nh=pd.read_csv(os.path.join(U,'tabaka_2020.csv')).set_index('_STSTR').n
def svy(dom,t):
    d=dom.astype(float); W=(wr*d).sum(); R=(wr*d*t).sum()/W; z=wr*d*(t-R)/W
    gdf=pd.DataFrame({'h':b._STSTR.values,'p':b._PSU.values,'z':z}).groupby(['h','p']).z.sum().reset_index().groupby('h').z.agg(s2=lambda s:(s**2).sum(),s1='sum')
    n=nh.reindex(gdf.index).values.astype(float); ok=n>1; var=(n[ok]/(n[ok]-1)*(gdf.s2.values[ok]-gdf.s1.values[ok]**2/n[ok])).sum(); se=np.sqrt(var)  # tek birimli tabakalar varyansa katkı vermez
    return R*100,max(0,(R-1.96*se)*100),min(100,(R+1.96*se)*100)
# Screening profiles
def bb(c,s): return c.astype(float).where(s.notna())
F=pd.DataFrame({'yas':XF.yas,'egitim':XF.egitim,'gelir':XF.gelir8,'sigortasiz':XF.sigortasiz,'doktor_yok':bb(XF.doktor==3,XF.doktor),'maliyet':bb(XF.maliyet_engeli==1,XF.maliyet_engeli),'checkup5':XF.checkup.replace({8:5}),
 'evli':bb(XF.medeni==1,XF.medeni),'hic_evlenmemis':bb(XF.medeni==5,XF.medeni),'cocuk':XF.cocuk,'kirsal':XF.kirsal,'ispanyolca':XF.dil_ispanyolca,'hispanik':bb(XF.irk==8,XF.irk),'siyah':bb(XF.irk==2,XF.irk),'asyali':bb(XF.irk==4,XF.irk),'beyaz':bb(XF.irk==1,XF.irk),
 'sigara_halen':bb(XF.sigara.isin([1,2]),XF.sigara),'ev_sahibi':bb(XF.ev_sahibi==1,XF.ev_sahibi),'calisiyor':bb(XF.istihdam.isin([1,2]),XF.istihdam)})
def tree(mask,target,leaf=2500):
    itr=np.intersect1d(tr,np.where(mask)[0]); t=DecisionTreeClassifier(max_depth=4,min_samples_leaf=leaf,random_state=0).fit(F.iloc[itr],target[itr],sample_weight=w[itr]); T=t.tree_
    lv=np.full(len(y),-1); lv[mask]=t.apply(F[mask]); base=np.average(target[mask],weights=w[mask]); tes=np.isin(np.arange(len(y)),te); trs=np.isin(np.arange(len(y)),tr)
    def node(n):
        if T.feature[n]==_tree.TREE_UNDEFINED:
            # The tree was derived on the training split. Report its apparent rate, share,
            # n and CI on that same derivation split; keep the held-out rate separate.
            k=(lv==n)&trs; r,lo,hi=svy(k,target.astype(float)); kt=(lv==n)&tes
            return dict(leaf=int(n),n=int(k.sum()),share=round(float(w[k].sum()/w[mask&trs].sum()*100),1),rate=round(r,1),lo=round(lo,1),hi=round(hi,1),rate_test=round(float(np.average(target[kt],weights=w[kt])*100),1))
        return dict(f=F.columns[T.feature[n]],t=float(T.threshold[n]),na_left=bool(T.missing_go_to_left[n]),l=node(T.children_left[n]),r=node(T.children_right[n]))
    return dict(base=round(float(base*100),1),root=node(0))
trees=dict(never=tree(np.ones(len(y),bool),(y==2).astype(int),1500),overdue=tree(y!=2,(y==1).astype(int)))
# Descriptive summaries
def grp(name,s,order=None):
    rows=[]
    for lvl in (order or sorted(s.dropna().unique())):
        k=(s==lvl).values
        if k.sum()==0: continue
        rows.append([str(lvl),int(k.sum())]+[round(float(np.average(y[k]==j,weights=w[k])*100),1) for j in range(3)])
    return rows
ag=pd.cut(XF.yas,[20,29,39,49,59,65],labels=['21–29','30–39','40–49','50–59','60–65']).astype(str)
desc=[dict(key='yas',label='Yaş grubu',rows=grp('',ag,['21–29','30–39','40–49','50–59','60–65'])),
 dict(key='sig',label='Sağlık sigortası',rows=grp('',XF.sigortasiz.map({0:'Sigortalı',1:'Sigortasız'}),['Sigortalı','Sigortasız'])),
 dict(key='dok',label='Düzenli doktor',rows=grp('',XF.doktor.map({1:'Doktoru var',2:'Doktoru var',3:'Düzenli doktoru yok'}),['Doktoru var','Düzenli doktoru yok'])),
 dict(key='mal',label='Maliyet engeli',rows=grp('',XF.maliyet_engeli.map({1:'Maliyet engeli var',2:'Maliyet engeli yok'}),['Maliyet engeli var','Maliyet engeli yok'])),
 dict(key='chk',label='Son check-up',rows=grp('',XF.checkup.map({1:'Son 1 yıl içinde',2:'1–2 yıl önce',3:'2–5 yıl önce',4:'5 yıldan eski',8:'Hiç'}),['Son 1 yıl içinde','1–2 yıl önce','2–5 yıl önce','5 yıldan eski','Hiç'])),
 dict(key='egt',label='Eğitim',rows=grp('',XF.egitim.map({1:'Lise altı',2:'Lise altı',3:'Lise altı',4:'Lise mezunu',5:'Üniversite terk / ön lisans',6:'Üniversite mezunu'}),['Lise altı','Lise mezunu','Üniversite terk / ön lisans','Üniversite mezunu'])),
 dict(key='gel',label='Hane geliri',rows=grp('',XF.gelir8.map({1:'< 15 bin $',2:'< 15 bin $',3:'15–25 bin $',4:'15–25 bin $',5:'25–35 bin $',6:'35–50 bin $',7:'50–75 bin $',8:'75 bin $ ve üzeri'}),['< 15 bin $','15–25 bin $','25–35 bin $','35–50 bin $','50–75 bin $','75 bin $ ve üzeri'])),
 dict(key='irk',label='Irk / etnisite',rows=grp('',XF.irk.map({1:'Beyaz',2:'Siyah',8:'Hispanik',4:'Asyalı',3:'Amerikan Yerlisi',5:'Pasifik Adalı',7:'Çok ırklı',6:'Diğer'}),['Beyaz','Siyah','Hispanik','Asyalı','Amerikan Yerlisi','Pasifik Adalı','Çok ırklı','Diğer'])),
 dict(key='med',label='Medeni durum',rows=grp('',XF.medeni.map({1:'Evli',2:'Boşanmış',3:'Dul',4:'Ayrı yaşıyor',5:'Hiç evlenmemiş',6:'Partneriyle yaşıyor'}))),
 dict(key='yer',label='Yerleşim',rows=grp('',XF.kirsal.map({0:'Kentsel',1:'Kırsal'}),['Kentsel','Kırsal'])),
 dict(key='sgr',label='Sigara',rows=grp('',XF.sigara.map({1:'Her gün içiyor',2:'Bazı günler içiyor',3:'Bırakmış',4:'Hiç içmemiş'}),['Her gün içiyor','Bazı günler içiyor','Bırakmış','Hiç içmemiş'])),
 dict(key='dil',label='Görüşme dili',rows=grp('',XF.dil_ispanyolca.map({0:'İngilizce',1:'İspanyolca'}),['İngilizce','İspanyolca']))]
fips={1:'Alabama',2:'Alaska',4:'Arizona',5:'Arkansas',6:'California',8:'Colorado',9:'Connecticut',10:'Delaware',11:'District of Columbia',12:'Florida',13:'Georgia',15:'Hawaii',16:'Idaho',17:'Illinois',18:'Indiana',19:'Iowa',20:'Kansas',21:'Kentucky',22:'Louisiana',23:'Maine',24:'Maryland',25:'Massachusetts',26:'Michigan',27:'Minnesota',28:'Mississippi',29:'Missouri',30:'Montana',31:'Nebraska',32:'Nevada',33:'New Hampshire',34:'New Jersey',35:'New Mexico',36:'New York',37:'North Carolina',38:'North Dakota',39:'Ohio',40:'Oklahoma',41:'Oregon',42:'Pennsylvania',44:'Rhode Island',45:'South Carolina',46:'South Dakota',47:'Tennessee',48:'Texas',49:'Utah',50:'Vermont',51:'Virginia',53:'Washington',54:'West Virginia',55:'Wisconsin',56:'Wyoming',66:'Guam',72:'Puerto Rico',78:'Virgin Islands'}
st=[[fips.get(int(s),str(int(s))),int(len(gx))]+[round(float(np.average(y[gx.index]==k,weights=w[gx.index])*100),1) for k in range(3)] for s,gx in b.groupby('_STATE')]
age=[[int(A)]+[round(float(np.average(y[gx.index]==k,weights=w[gx.index])*100),1) for k in range(3)] for A,gx in b.groupby('_AGE80')]
age24=[[int(A)]+[round(float(np.average(ya[gx.index]==k,weights=wa[gx.index])*100),1) for k in range(3)] for A,gx in a.groupby('_AGE80')]
cls=[[n_,int((y==k).sum()),round(float(np.average(y==k,weights=w)*100),1)] for k,n_ in enumerate(['Güncel','Geri kalmış','Hiç taranmamış'])]
d0=pd.read_csv(os.path.join(U,'brfss2020_kadin.csv.gz'),usecols=['_AGE80','HADHYST2']); 
flow=[['BRFSS 2020, tüm görüşmeler',int(nh.sum())],['Kadınlar',int(len(d0))],['21–65 yaş',int(((d0._AGE80>=21)&(d0._AGE80<=65)).sum())],['Histerektomi geçirmemiş',int(((d0._AGE80>=21)&(d0._AGE80<=65)&(d0.HADHYST2==2)).sum())],['Tarama bilgisi tam (analitik örneklem)',int(len(y))]]
# Browser-model export
coef=lr.coef_; groups={}
for c in Z.columns: groups.setdefault('yas' if c.startswith('yas') else c.split('=')[0],[]).append(c)
means={gk:[float(np.average(Z[cs].values@(coef[k,[Z.columns.get_loc(c) for c in cs]]-coef[0,[Z.columns.get_loc(c) for c in cs]]),weights=w)) for k in (1,2)] for gk,cs in groups.items()}
tests=[]
for i in list(te[:6])+[int(np.argmax(ph[:,2]))]:
    o={'yas':float(XF.yas.iloc[i]),'cocuk':float(XF.cocuk.fillna(0).iloc[i])}
    for v,(l,lv) in V.items():
        x=XF[v].iloc[i]; o[v]=int(x) if (x==x and x in lv) else None
    tests.append(dict(o=o,p=np.round(lr.predict_proba(Z.iloc[[i]])[0],4).tolist()))
cv=pd.read_csv('out/final_cv.csv').round(4).to_dict('records')
model=dict(cols=list(Z.columns),coef=np.round(coef,5).tolist(),intercept=np.round(lr.intercept_,5).tolist(),knots=knots,means=means,vars={v:dict(label=l,levels=[[int(c),t] for c,t in lv.items()]) for v,(l,lv) in V.items()})
json.dump(dict(model=model,cal=cal,tests=tests),open('out/webmodel.json','w'),ensure_ascii=False)
json.dump(dict(trees=trees,desc=desc,states=st,age=age,age24=age24,flow=flow,classes=cls,cv=cv,internal=internal,ext=ext,lift=lift,sub=sub,lgbm=lgbm,shap=shap_out,base=[c[2] for c in cls],ntest=int(len(te)),n24=int(len(ya))),open('out/webdata.json','w'),ensure_ascii=False)
print('flow',flow); print('cls',cls); print('BITTI')
