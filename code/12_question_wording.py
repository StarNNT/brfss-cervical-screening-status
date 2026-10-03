# Question-wording analysis: never-screened share, don't-know share, age pattern, birth-cohort comparison and state spread under the three
# instruments (2020 Pap item; 2022 bare "cervical cancer screening test" item; 2024 item with definitional preamble). Writes out/wording.json.
import pandas as pd, numpy as np, json, os
U=os.environ.get('BRFSS_DATA_DIR','.'); out={}
d20=pd.read_csv(os.path.join(U,'brfss2020_kadin.csv.gz'),usecols=['_AGE80','HADPAP2','HADHYST2','_LLCPWT','_STATE']); w20=d20[(d20._AGE80.between(21,65))&(d20.HADHYST2==2)]
def summ(w,ever,valid):
    r={'n':int(len(w)),'dk_pct':round(float(np.average(~valid,weights=w._LLCPWT))*100,1)}; v=w[valid]; nv=(ever[valid]==2)
    r['never_pct']=round(float(np.average(nv,weights=v._LLCPWT))*100,1)
    ag=pd.cut(v._AGE80,[20,29,39,49,59,65],labels=['21-29','30-39','40-49','50-59','60-65'])
    r['never_by_age']={str(k):round(float(np.average(nv[ag==k],weights=v._LLCPWT[ag==k]))*100,1) for k in ag.cat.categories}
    g=v.assign(nv=nv).groupby('_STATE').apply(lambda x: np.average(x.nv,weights=x._LLCPWT)*100)
    r['state_min']=round(float(g.min()),1); r['state_max']=round(float(g.max()),1); r['state_median']=round(float(g.median()),1); r['n_states']=int(len(g)); return r,g
out['2020'],g20=summ(w20,w20.HADPAP2,w20.HADPAP2.isin([1,2]))
W={}
for y in [2022,2024]:
    d=pd.read_csv(os.path.join(U,f'brfss{y}_tarama.csv.gz')); w=d[(d._SEX==2)&(d._AGE80.between(21,65))&(d.HADHYST2==2)]; out[str(y)],W[y]=summ(w,w.CERVSCRN,w.CERVSCRN.isin([1,2])); W[y]=(w[w.CERVSCRN.isin([1,2])],W[y])
def coh(w,ever,a,c): m=w._AGE80.between(a,c); return round(float(np.average(ever[m]==2,weights=w._LLCPWT[m]))*100,1)
v20=w20[w20.HADPAP2.isin([1,2])]
out['cohort']={'born_1965_1975':{'2020_age45_55':coh(v20,v20.HADPAP2,45,55),'2022_age47_57':coh(W[2022][0],W[2022][0].CERVSCRN,47,57),'2024_age49_59':coh(W[2024][0],W[2024][0].CERVSCRN,49,59)},
 'born_1980_1990':{'2020_age30_40':coh(v20,v20.HADPAP2,30,40),'2022_age32_42':coh(W[2022][0],W[2022][0].CERVSCRN,32,42),'2024_age34_44':coh(W[2024][0],W[2024][0].CERVSCRN,34,44)}}
j=pd.concat([W[2022][1],W[2024][1]],axis=1,keys=['y22','y24']).dropna(); out['state_corr_2022_2024']=round(float(j.y22.corr(j.y24)),2); out['state_ratio_2022_2024']={'median':round(float((j.y22/j.y24).median()),2),'min':round(float((j.y22/j.y24).min()),2),'max':round(float((j.y22/j.y24).max()),2)}
json.dump(out,open('out/wording.json','w'),indent=1); print(json.dumps(out,indent=1))
