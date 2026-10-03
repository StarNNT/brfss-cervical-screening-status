import pandas as pd, numpy as np
NUM=['yas','cocuk','fiziksel_kotu_gun','ruhsal_kotu_gun','bmi']
def feats(c,year):
    n=(lambda a,b: a if year==2024 else b)
    X=pd.DataFrame(index=c.index)
    def cat(v,miss=(7,9)):
        s=c[v].copy(); s[s.isin(miss)]=np.nan; return s
    X['yas']=c._AGE80; X['irk']=cat('_RACE',(9,)); X['egitim']=cat('EDUCA',(9,))
    if year==2024: g=cat('INCOME3',(77,99)).clip(upper=8)
    else: g=cat('INCOME2',(77,99))
    X['gelir8']=g
    X['medeni']=cat('MARITAL',(9,)); X['istihdam']=cat('EMPLOY1',(9,))
    ch=c.CHILDREN.copy(); ch[ch==88]=0; ch[ch==99]=np.nan; X['cocuk']=ch.clip(upper=5)
    X['ev_sahibi']=cat('RENTHOM1'); X['kirsal']=cat('_URBSTAT')-1; X['dil_ispanyolca']=(c.QSTLANG==2).astype(float); X['gazi']=cat('VETERAN3'); X['eyalet']=c._STATE
    if year==2024:
        p=cat('PRIMINS2',(77,99)); X['sigortasiz']=(p==88).astype(float).where(p.notna())
    else:
        p=cat('HLTHPLN1'); X['sigortasiz']=(p==2).astype(float).where(p.notna())
    X['doktor']=cat(n('PERSDOC3','PERSDOC2')); X['maliyet_engeli']=cat(n('MEDCOST1','MEDCOST')); X['checkup']=cat('CHECKUP1'); X['genel_saglik']=cat('GENHLTH')
    for v,nm in [('PHYSHLTH','fiziksel_kotu_gun'),('MENTHLTH','ruhsal_kotu_gun')]:
        s=c[v].copy(); s[s==88]=0; s[s.isin([77,99])]=np.nan; X[nm]=s
    X['bmi']=c._BMI5/100; X['sigara']=cat('_SMOKER3',(9,)); X['asiri_icme']=cat(n('_RFBING6','_RFBING5'),(9,)); X['egzersiz']=cat('EXERANY2')
    X['grip_asisi']=cat('FLUSHOT7'); X['hiv_testi']=cat('HIVTST7'); X['dis_hekimi']=cat('LASTDEN4',(7,9))
    for v,nm in [('ADDEPEV3','depresyon'),('DIABETE4','diyabet'),('HAVARTH4','artrit'),('ASTHMA3','astim'),(n('CHCCOPD3','CHCCOPD2'),'koah'),('_MICHD','kalp_hast'),
                ('DIFFWALK','yurume_guclugu'),('DECIDE','bilissel_gucluk'),('DIFFALON','yalniz_is_guclugu'),('DEAF','isitme'),('BLIND','gorme')]: X[nm]=cat(v)
    return X
