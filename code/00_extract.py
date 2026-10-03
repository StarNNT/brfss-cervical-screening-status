# Reads a BRFSS SAS transport file in chunks; keeps women and the study variables.
# Usage: python code/00_extract.py 2020   (expects LLCP2020.XPT in the repository root; same for 2024)
# Output: brfss<year>_kadin.csv.gz and, for 2020, tabaka_2020.csv (records per stratum, needed for design-based CIs)
import pandas as pd, sys
y=int(sys.argv[1])
K={2020:['CVDINFR4','CVDCRHD4','CVDSTRK3','CHCSCNCR','CHCKDNY2','DIFFDRES','SLEPTIM1','_RFDRHV7','SEATBELT','HIVRISK5','HADMAM','HOWLONG','HPVADVC4','HPVADSHT','SOFEMALE','TRNSGNDR','ACEDEPRS','ACEDRINK','ACEDRUGS','ACEPRISN','ACEDIVRC','ACEPUNCH','ACEHURT1','ACESWEAR','ACETOUCH','ACETTHEM','ACEHVSEX','HLTHCVR1','NUMADULT','HHADULT','CPDEMO1B','ECIGNOW','MARIJAN1','TETANUS1','CAREGIV1','_STATE','_PSU','_STSTR','_LLCPWT','_SEX','_AGE80','GENHLTH','PHYSHLTH','MENTHLTH','HLTHPLN1','PERSDOC2','MEDCOST','CHECKUP1','EXERANY2','LASTDEN4','ASTHMA3','CHCOCNCR','CHCCOPD2','ADDEPEV3','HAVARTH4','DIABETE4','MARITAL','EDUCA','RENTHOM1','VETERAN3','EMPLOY1','CHILDREN','INCOME2','PREGNANT','DEAF','BLIND','DECIDE','DIFFWALK','DIFFALON','HADPAP2','LASTPAP2','HPVTEST','HPLSTTST','HADHYST2','_SMOKER3','_RFBING5','FLUSHOT7','HIVTST7','_URBSTAT','_RACE','_BMI5','_MICHD','QSTLANG','_RFPAP35'],
   2024:['_STATE','_PSU','_STSTR','_LLCPWT','_SEX','_AGE80','GENHLTH','PHYSHLTH','MENTHLTH','PRIMINS2','PERSDOC3','MEDCOST1','CHECKUP1','EXERANY2','LASTDEN4','ASTHMA3','CHCOCNC1','CHCCOPD3','ADDEPEV3','HAVARTH4','DIABETE4','MARITAL','EDUCA','RENTHOM1','VETERAN3','EMPLOY1','CHILDREN','INCOME3','PREGNANT','DEAF','BLIND','DECIDE','DIFFWALK','DIFFALON','CERVSCRN','CRVCLCNC','CRVCLPAP','CRVCLHPV','HADHYST2','_SMOKER3','_RFBING6','FLUSHOT7','HIVTST7','_URBSTAT','_RACE','_BMI5','_MICHD','QSTLANG','_PAPHPV1']}[y]
parts=[];st=[]
for ch in pd.read_sas(f'LLCP{y}.XPT',format='xport',chunksize=50000):
    st.append(ch.groupby('_STSTR').size()); parts.append(ch.loc[ch['_SEX']==2,[c for c in K if c in ch.columns]])
pd.concat(parts).to_csv(f'brfss{y}_kadin.csv.gz',index=False)
pd.concat(st).groupby(level=0).sum().rename('n').to_csv(f'tabaka_{y}.csv')
