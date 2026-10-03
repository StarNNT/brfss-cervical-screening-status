# "Sınıftan kadına": her tarama sınıfının bileşimi, fazla temsil edilen özellikler ve sınıf içi alt tipler (ağırlıklı k-ortalamalar)
import pandas as pd, numpy as np, sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'out'); sys.path.insert(0,'code')
from harmon import feats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
b=pd.read_pickle('out/serviks2020.pkl').reset_index(drop=True); y=b.y.astype(int).values; w=b._LLCPWT.values; X=feats(b,2020)
G={'21–25 yaş':X.yas<=25,'26–39 yaş':(X.yas>25)&(X.yas<40),'40–54 yaş':(X.yas>=40)&(X.yas<55),'55–65 yaş':X.yas>=55,
'Evli':X.medeni==1,'Hiç evlenmemiş':X.medeni==5,'Boşanmış, dul ya da ayrı':X.medeni.isin([2,3,4]),'Çocuklu hane':X.cocuk>=1,
'Çalışıyor':X.istihdam.isin([1,2]),'Öğrenci':X.istihdam==6,'İşsiz':X.istihdam.isin([3,4]),'Ev hanımı':X.istihdam==5,'Çalışamıyor':X.istihdam==8,
'Lise veya altı':X.egitim<=4,'Üniversite mezunu':X.egitim==6,'Gelir < 35 bin $':X.gelir8<=5,'Gelir ≥ 75 bin $':X.gelir8==8,'Kiracı':X.ev_sahibi==2,
'Beyaz':X.irk==1,'Siyah':X.irk==2,'Hispanik':X.irk==8,'Asyalı':X.irk==4,'İspanyolca görüşme':X.dil_ispanyolca==1,'Kırsal':X.kirsal==1,
'Sigortasız':X.sigortasiz==1,'Düzenli doktoru yok':X.doktor==3,'Maliyet yüzünden doktora gidememiş':X.maliyet_engeli==1,'Son check-up 2 yıldan eski ya da hiç':X.checkup.isin([3,4,8]),
'Diş hekimine 2+ yıldır gitmemiş':X.dis_hekimi.isin([3,4,8]),'Grip aşısı olmamış':X.grip_asisi==2,'Hiç HIV testi yaptırmamış':X.hiv_testi==2,'Halen sigara içiyor':X.sigara.isin([1,2]),'Sağlığı orta ya da kötü':X.genel_saglik>=4}
F=pd.DataFrame({k:v.values.astype(float) for k,v in G.items()})
allp=(F.values*w[:,None]).sum(0)/w.sum()*100
def wmed(v,ww):
    o=np.argsort(v); c=np.cumsum(ww[o]); return float(v[o][np.searchsorted(c,c[-1]/2)])
out={}
for k,name in [(2,'never'),(1,'overdue'),(0,'current')]:
    m=y==k; Fk=F[m].values; wk=w[m]; cp=(Fk*wk[:,None]).sum(0)/wk.sum()*100
    comp=[[f,round(float(a),1),round(float(c),1),round(float(c/a),2)] for f,a,c in zip(F.columns,allp,cp)]
    # alt tipler
    Z=(Fk-Fk.mean(0))/(Fk.std(0)+1e-9); best=None
    rs=np.random.RandomState(0); sub=rs.choice(len(Z),min(6000,len(Z)),replace=False)
    for K in (3,4,5):
        km=KMeans(K,n_init=10,random_state=0).fit(Z,sample_weight=wk); s=silhouette_score(Z[sub],km.labels_[sub]); print(name,K,round(s,3),flush=True)
        if best is None or s>best[0]+0.005: best=(s,K,km)
    s,K,km=best; cl=[]
    for c in range(K):
        mm=km.labels_==c; sh=wk[mm].sum()/wk.sum()*100; p=(Fk[mm]*wk[mm][:,None]).sum(0)/wk[mm].sum()*100
        d=sorted(zip(F.columns,p,cp),key=lambda t:-(t[1]-t[2]))
        cl.append(dict(share=round(float(sh),1),n=int(mm.sum()),yas=wmed(X.yas.values[m][mm],wk[mm]),top=[[f,round(float(a),1),round(float(bb),1)] for f,a,bb in d[:7]],low=[[f,round(float(a),1),round(float(bb),1)] for f,a,bb in d[-3:]]))
    cl.sort(key=lambda c:-c['share'])
    out[name]=dict(n=int(m.sum()),share=round(float(wk.sum()/w.sum()*100),1),yas=wmed(X.yas.values[m],wk),comp=comp,clusters=cl,K=K,sil=round(float(s),3))
    print('==',name,out[name]['n'],out[name]['share'],'K',K)
    for c in cl: print('  pay',c['share'],'n',c['n'],'yaş',c['yas'],'|',', '.join(f"{f} {a:.0f}% (sınıf {bb:.0f}%)" for f,a,bb in c['top'][:6]))
# Cluster count can change when the audited outcome definition changes. Use neutral,
# deterministic names instead of hand-authored demographic labels that may stereotype
# a cluster or silently become attached to the wrong solution.
for k in out:
    for i,c in enumerate(out[k]['clusters'],1): c['name']=f'Keşifsel profil {i}'
W=json.load(open('out/webdata.json')); W['rev']=out; json.dump(W,open('out/webdata.json','w'),ensure_ascii=False)
