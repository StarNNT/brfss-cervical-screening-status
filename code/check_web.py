"""Static and numerical checks for the generated Turkish and English web artifacts."""
from pathlib import Path
import json, math, re, sys

ROOT=Path(__file__).resolve().parents[1]
errors=[]
release='--release' in sys.argv

def fail(message): errors.append(message)

def embedded(html):
    hit=re.search(r'const M=(\{.*?\}), W=(\{.*?\}), CAL=(\[.*?\]);',html,re.S)
    if not hit: raise ValueError('embedded model/data/calibration payload not found')
    return json.loads(hit.group(1)),json.loads(hit.group(2)),json.loads(hit.group(3))

def predict(model,o):
    cols=model['cols']; ix={c:i for i,c in enumerate(cols)}; x={}
    x['yas']=o['yas']-21
    for k in model['knots']: x['yas_k'+str(k)]=max(0,o['yas']-k)
    x['cocuk']=o.get('cocuk') if o.get('cocuk') is not None else model.get('colmeans',{}).get('cocuk',0)
    for v in model['vars']:
        if o.get(v) is None:
            for c in cols:
                if c.startswith(v+'='): x[c]=model.get('colmeans',{}).get(c,0)
        else: x[v+'='+str(o[v])]=1
    z=list(model['intercept'])
    for c,v in x.items():
        if c not in ix: continue
        j=ix[c]
        for k in range(3): z[k]+=model['coef'][k][j]*v
    mx=max(z); e=[math.exp(v-mx) for v in z]; den=sum(e)
    return [v/den for v in e]

payloads=[]
for name,lang in [('tr.html','tr'),('index.html','en')]:
    path=ROOT/name
    if not path.exists(): fail(name+' is missing'); continue
    s=path.read_text(encoding='utf-8')
    if not s.startswith('<!doctype html>'): fail(name+' has no HTML5 doctype')
    for tag in ('html','head','body'):
        if len(re.findall(r'<'+tag+r'\b',s,re.I))!=1: fail(name+f' must contain exactly one <{tag}>')
    if f'<html lang="{lang}">' not in s: fail(name+' has the wrong lang attribute')
    ids=re.findall(r'\bid="([^"]+)"',s); dup=sorted({x for x in ids if ids.count(x)>1})
    if dup: fail(name+' has duplicate IDs: '+', '.join(dup))
    external=re.findall(r'<(?:script|link|img)\b[^>]*(?:src|href)="https?://',s,re.I)
    if external: fail(name+' loads external scripts, styles, or images')
    for old in ('Yüksek riskli','Düşük riskli','Risk hesaplayıcı','Serviks Tarama Risk Aracı'):
        if old in s: fail(name+' retains unsafe/obsolete wording: '+old)
    try: payloads.append((name,)+embedded(s))
    except Exception as exc: fail(name+': '+str(exc))

if len(payloads)==2:
    tr_name,tr_m,tr_w,tr_cal=payloads[0]; en_name,en_m,en_w,en_cal=payloads[1]
    if tr_m['cols']!=en_m['cols'] or tr_m['coef']!=en_m['coef']: fail('TR and EN model payloads differ')
    if tr_cal!=en_cal: fail('TR and EN calibration payloads differ')
    if len(tr_m['coef'])!=3 or any(len(row)!=len(tr_m['cols']) for row in tr_m['coef']): fail('model coefficient dimensions are invalid')
    if len(tr_w.get('fairness',[]))<10: fail('subgroup decision-rate metrics are missing')
    if not tr_w.get('artifact_meta',{}).get('version'): fail('artifact version metadata is missing')
    if release and tr_w.get('artifact_meta',{}).get('source')!='reproduced_outputs': fail('release check refuses coefficients inherited from an older embedded artifact')
    sample={'yas':34,'cocuk':2,'irk':8,'egitim':4,'gelir8':5,'medeni':6,'istihdam':1,'ev_sahibi':2,'kirsal':0,'dil_ispanyolca':1,
            'sigortasiz':1,'doktor':3,'maliyet_engeli':1,'checkup':3,'genel_saglik':3,'sigara':4,'grip_asisi':2,'hiv_testi':2,'dis_hekimi':3}
    sparse={v:None for v in tr_m['vars']}; sparse.update({'yas':40,'cocuk':None})
    for label,case in [('complete sample',sample),('age-only sample',sparse)]:
        prob=predict(tr_m,case)
        if not all(math.isfinite(x) and 0<=x<=1 for x in prob) or abs(sum(prob)-1)>1e-9: fail(label+' does not produce valid probabilities')
    for target in ('never','overdue'):
        rows=tr_w['thr'][target]['rows']
        if any(rows[i][1]>rows[i-1][1]+0.11 for i in range(1,len(rows))): fail(target+' flagged share is not monotone by threshold')
        if any(rows[i][2]>rows[i-1][2]+0.11 for i in range(1,len(rows))): fail(target+' sensitivity is not monotone by threshold')
        if any(rows[i][3]<rows[i-1][3]-0.11 for i in range(1,len(rows))): fail(target+' specificity is not monotone by threshold')

tests_path=ROOT/'out'/'tests.json'
if tests_path.exists() and payloads:
    tests=json.loads(tests_path.read_text(encoding='utf-8'))
    for i,test in enumerate(tests):
        got=predict(payloads[0][1],test['o'])
        if max(abs(a-b) for a,b in zip(got,test['p']))>5e-4: fail(f'numerical browser-model test {i} failed')

audit_path=ROOT/'out'/'label_audit.json'
if audit_path.exists():
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['2020'].get('never_with_unknown_hpv')!=0: fail('unknown/refused HPV response was assigned to never screened')

template=(ROOT/'web_src'/'template.html').read_text(encoding='utf-8')
if 'setForm(SAMPLE);update()' in template: fail('the page still auto-loads the sample case')
for required in ('setForm(null);update()','if(v==null)return null','result-empty','artifact-warning'):
    if required not in template: fail('required UI safeguard is missing: '+required)

if errors:
    for e in errors: print('FAIL',e)
    sys.exit(1)
print('OK  static HTML, bilingual payload, privacy, wording, safeguards, and numerical tests'+(' (release mode)' if release else ''))
