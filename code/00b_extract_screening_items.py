# Reads LLCP2021.XPT ... LLCP2024.XPT in chunks and keeps the cervical, breast and colorectal screening items plus the study variables for all respondents.
# Usage: python code/00b_extract_screening_items.py 2022   (expects LLCP<year>.XPT in the repository root)
# Output: brfss<year>_tarama.csv.gz (used by code/12_question_wording.py and code/supplementary/multi_year_2021_2024.py)
import pandas as pd, sys
y=int(sys.argv[1])
K=['_STATE','IYEAR','DISPCODE','QSTVER','QSTLANG','_PSU','_STSTR','_LLCPWT','SEXVAR','_SEX','_AGE80','_AGEG5YR','GENHLTH','PHYSHLTH','MENTHLTH','PRIMINS2','PERSDOC3','MEDCOST1','CHECKUP1','EXERANY2','LASTDEN4','CVDINFR4','CVDCRHD4','CVDSTRK3','ASTHMA3','CHCSCNC1','CHCOCNC1','CHCCOPD3','ADDEPEV3','CHCKDNY2','HAVARTH4','DIABETE4','MARITAL','EDUCA','RENTHOM1','VETERAN3','EMPLOY1','CHILDREN','INCOME3','PREGNANT','DEAF','BLIND','DECIDE','DIFFWALK','DIFFDRES','DIFFALON','HADMAM','HOWLONG','CERVSCRN','CRVCLCNC','CRVCLPAP','CRVCLHPV','HADHYST2','SMOKE100','_SMOKER3','ECIGNOW3','ALCDAY4','_RFBING6','_RFDRHV9','FLUSHOT7','HIVTST7','HPVADVC4','_METSTAT','_URBSTAT','_IMPRACE','_RACE','_RACEGR3','_HISPANC','_BMI5','_BMI5CAT','_CHLDCNT','_EDUCAG','_INCOMG1','_HLTHPL2','_TOTINDA','_MICHD','_CRVSCRN','_RFPAP37','_HPV5YR1','_PAPHPV1','_RFMAM23','HADSIGM4','COLNSIGM','COLNTES1','SIGMTES1','LASTSIG4','COLNCNCR','VIRCOLO1','VCLNTES2','SMALSTOL','STOLTEST','STOOLDN2','BLDSTFIT','SDNATES1','_CRCREC2','_CRCREC3','_HADCOLN','_CLNSCP2','_HADSIGM','_SGMSCP2','_SGMS102','_RFBLDS6','_STOLDN2','_VIRCOL2','_SBONTI2','_MAM402Y','LSATISFY','EMTSUPRT','SDLONELY','SDHEMPLY','FOODSTMP','SDHFOOD1','SDHBILLS','SDHUTILS','SDHTRNSP','SOFEMALE']
parts=[]
for ch in pd.read_sas(f'LLCP{y}.XPT',format='xport',chunksize=50000):
    parts.append(ch[[c for c in K if c in ch.columns]])
d=pd.concat(parts); d['YIL']=y; d.to_csv(f'brfss{y}_tarama.csv.gz',index=False); print(y,len(d))
