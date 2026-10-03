"""Check consistency across outputs from a complete analysis run.

Reference values should be updated only after the revised outcome definition has been
used throughout the analysis and the resulting outputs have been reviewed.
"""
import json, math, sys

W=json.load(open('out/webdata.json'))
H=json.load(open('out/model_comparison.json'))
R=json.load(open('out/design_bootstrap.json'))
Q=json.load(open('out/wording.json'))
A=json.load(open('out/label_audit.json'))
errors=[]

def check(ok,message):
    print(('OK  ' if ok else 'FAIL'),message)
    if not ok: errors.append(message)

check(A['2020']['never_with_unknown_hpv']==0,'never-screened requires an explicit HPV No response')
check(A['2020']['eligible']==A['2020']['analytic']+A['2020']['excluded_ambiguous'],'2020 label audit accounts for every eligible record')
check(sum(A['2020']['counts'].values())==A['2020']['analytic'],'2020 class counts equal analytic n')
check(sum(A['2024']['counts'].values())==A['2024']['analytic'],'2024 class counts equal analytic n')
check(sum(c[1] for c in W['classes'])==A['2020']['analytic'],'web class counts match the corrected analytic sample')

for where,obj in [('internal',W['internal']),('2024',W['ext'])]:
    for key in ('guncel','geri','hic','geri_taranmislar'):
        est,lo,hi=obj[key]
        check(0.5<=est<=1 and 0<=lo<=est<=hi<=1,f'{where} {key} AUC and CI are ordered')
    check(math.isfinite(obj['logloss']) and obj['logloss']>0,f'{where} log-loss is finite and positive')

mapping={'guncel':'current','geri':'overdue','hic':'never'}
for web_key,rev_key in mapping.items():
    b=R['boot_internal']['auc_lr_'+rev_key]; a=W['internal'][web_key]
    check(max(abs(a[i]-[b['est'],b['lo'],b['hi']][i]) for i in range(3))<0.002,f'web internal {web_key} CI matches design bootstrap')

for target in ('never','overdue'):
    default=R['thresholds']['default'][target]
    check(0<default<1,f'{target} default threshold is a probability')
    rows=W['thr'][target]['rows']; d=W['thr'][target]['default']
    check(1<=d<=len(rows) and rows[d-1][0]==d,f'{target} threshold table default indexes the matching row')

check(0<=Q['2022']['never_pct']<=100 and 0<=Q['2024']['never_pct']<=100,'question-wording percentages are valid')
check('external_2024' in H['holdout']['LightGBM (tuned)'],'tuned-model 2024 evaluation is present')

if errors: sys.exit(1)
