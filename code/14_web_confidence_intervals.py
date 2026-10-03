# Replaces the AUC confidence intervals used by the web page (out/webdata.json, written by script 03 with a simple iid bootstrap) with the design-based
# bootstrap intervals from script 13 (out/revizyon.json, 1,000 within-stratum resamples), so that the page and the paper report the same intervals. Run after 13, before the web build.
# Web aracındaki AUC güven aralıklarını (webdata.json) tasarım temelli bootstrap (revizyon.json, B=1000) ile değiştirir.
import json
W=json.load(open('out/webdata.json')); R=json.load(open('out/revizyon.json'))
M={'hic':'auc_lr_never','geri':'auc_lr_overdue','guncel':'auc_lr_current','geri_taranmislar':'auc_lr_overdue_among_screened'}
for sec,bk in [('internal','boot_internal'),('ext','boot_external_2024')]:
    for k,rk in M.items():
        r=R[bk][rk]; old=W[sec][k]; W[sec][k]=[round(r['est'],3),round(r['lo'],3),round(r['hi'],3)]; print(sec,k,old,'->',W[sec][k])
    W[sec]['logloss']=round(R[bk]['logloss_lr']['est'],3)
W['ci_note']='95% CI: design-based bootstrap, 1,000 within-stratum resamples (revizyon.json)'
W['ntest']=W.get('ntest',19770); json.dump(W,open('out/webdata.json','w'),ensure_ascii=False,indent=1)
