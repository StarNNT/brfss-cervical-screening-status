# Eligibility (women 21-65, no hysterectomy) and the three-class outcome for both survey years.
# y: 0 = up to date, 1 = overdue, 2 = never screened
import pandas as pd, numpy as np, os, json
os.makedirs('out',exist_ok=True)
DATA=os.environ.get('BRFSS_DATA_DIR','.')
audit={}
# --- 2020: separate Pap and HPV questions. Some local extraction archives keep
# optional modules in a superset file; prefer it when present so later analyses
# can reuse the same audited analytic pickle without reimplementing the labels.
p20=os.path.join(DATA,'brfss2020_kadin_ek.csv.gz')
if not os.path.exists(p20): p20=os.path.join(DATA,'brfss2020_kadin.csv.gz')
d=pd.read_csv(p20); b=d[(d._AGE80>=21)&(d._AGE80<=65)&(d.HADHYST2==2)].copy()
pap_known=(b.HADPAP2==1)&b.LASTPAP2.isin([1,2,3,4,5])
hpv_known=(b.HPVTEST==1)&b.HPLSTTST.isin([1,2,3,4,5])
pap3=pap_known&b.LASTPAP2.isin([1,2,3])
hpv5=(b._AGE80>=30)&hpv_known&b.HPLSTTST.isin([1,2,3,4])
# A "never" label requires explicit No answers to both ever-screened items. Unknown/refused
# HPV answers are not silently converted to No. A known prior Pap or HPV test that is not
# current is overdue; truly ambiguous histories remain outside the analytic sample.
y=pd.Series(np.nan,index=b.index)
y[(b.HADPAP2==2)&(b.HPVTEST==2)]=2
y[pap_known|hpv_known]=1
y[pap3|hpv5]=0
audit['2020']={'eligible':int(len(b)),'analytic':int(y.notna().sum()),'excluded_ambiguous':int(y.isna().sum()),
               'never_with_unknown_hpv':int(((y==2)&~b.HPVTEST.eq(2)).sum()),
               'counts':{str(int(k)):int(v) for k,v in y.value_counts().sort_index().items()}}
b['y']=y; b=b[b.y.notna()]; b.to_pickle('out/serviks2020.pkl'); print(2020,len(b),b.y.value_counts().to_dict())
# --- 2024: single "cervical cancer screening test" question
d=pd.read_csv(os.path.join(DATA,'brfss2024_kadin.csv.gz')); a=d[(d._AGE80>=21)&(d._AGE80<=65)&(d.HADHYST2==2)].copy()
ok=(a.CERVSCRN==1)&a.CRVCLCNC.isin([1,2,3,4,5]); cur=ok&((a.CRVCLCNC<=3)|((a._AGE80>=30)&(a.CRVCLCNC==4)&(a.CRVCLHPV==1)))
y=pd.Series(np.nan,index=a.index,dtype=object); y[ok]='overdue'; y[a.CERVSCRN==2]='never'; y[cur]='current'
audit['2024']={'eligible':int(len(a)),'analytic':int(y.notna().sum()),'excluded_ambiguous':int(y.isna().sum()),
               'counts':{str(k):int(v) for k,v in y.value_counts().items()}}
a['y']=y; a=a[a.y.notna()]; a.to_pickle('out/analitik.pkl'); print(2024,len(a),a.y.value_counts().to_dict())
json.dump(audit,open('out/label_audit.json','w'),ensure_ascii=False,indent=2)
