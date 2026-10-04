# Build the English and Turkish static pages from the exported analysis results.
import json,copy,re,os

def load_inputs():
    """Build only from the current, fully regenerated analysis outputs."""
    required=['../out/webmodel.json','../out/webdata.json']
    missing=[p for p in required if not os.path.exists(p)]
    if missing: raise FileNotFoundError('missing current analysis output(s): '+', '.join(missing))
    model=json.load(open('../out/webmodel.json',encoding='utf-8'))
    data=json.load(open('../out/webdata.json',encoding='utf-8'))
    source='analysis_outputs'
    for key,name in [('blocks','ek_bloklar.json'),('stress','stres.json')]:
        path=os.path.join('../out',name)
        if os.path.exists(path): data[key]=json.load(open(path,encoding='utf-8'))
        if key not in data: raise FileNotFoundError(path)
    data.pop('build_meta',None)
    data['build_meta']={'version':'1.2.0','built':'2026-10-04','source':source,'development':'BRFSS 2020','evaluation':'BRFSS 2024'}
    return model,data

def add_fairness(data):
    """Attach decision-rate metrics when the design-bootstrap output is available."""
    paths=['../out/design_bootstrap.json']
    path=next((p for p in paths if os.path.exists(p)),None)
    if not path: return
    boot=json.load(open(path,encoding='utf-8')).get('boot_internal',{})
    groups=[
      ('White, non-Hispanic','Beyaz (Hispanik olmayan)','White, non-Hispanic'),('Black, non-Hispanic','Siyah (Hispanik olmayan)','Black, non-Hispanic'),
      ('Hispanic','Hispanik','Hispanic'),('Asian','Asyalı','Asian'),('AI/AN','Amerikan Yerlisi / Alaska Yerlisi','AI/AN'),('Multiracial/other','Çok ırklı / diğer','Multiracial/other'),
      ('Age 21-29','21–29 yaş','Age 21–29'),('Age 30-49','30–49 yaş','Age 30–49'),('Age 50-65','50–65 yaş','Age 50–65'),
      ('Insured','Sigortalı','Insured'),('Uninsured','Sigortasız','Uninsured'),('Urban','Kentsel','Urban'),('Rural','Kırsal','Rural'),
      ('English interview','İngilizce görüşme','English interview'),('Spanish interview','İspanyolca görüşme','Spanish interview')]
    def val(name,scale=1):
        x=boot.get(name)
        return None if not x else [round(float(x[k])*scale,3) for k in ('est','lo','hi')]
    rows=[]
    for key,tr,en in groups:
        row={'key':key,'label':tr,'label_en':en,'auc_never':val('g_auc_never_'+key),
             'sensitivity':val('g_sens_'+key),'ppv':val('g_ppv_'+key),'fpr':val('g_fpr_'+key),'alert':val('g_alert_'+key),
             'observed':val('g_obs_never_'+key,100),'predicted':val('g_pred_never_'+key,100)}
        if row['sensitivity'] is not None: rows.append(row)
    if rows: data['fairness']=rows

m,w=load_inputs(); add_fairness(w)
js=lambda o: json.dumps(o,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
def build(tpl,model,data,out,lang):
    t=open(tpl,encoding='utf-8').read().replace('/*MODEL*/',js(model)).replace('/*DATA*/',js(data)).replace('/*CAL*/',js(m['cal']))
    other=('tr.html','Türkçe') if lang=='en' else ('index.html','English')
    t=t.replace('<header class="top">','<header class="top"><a class="btn" style="align-self:flex-end;text-decoration:none" href="'+other[0]+'">'+other[1]+'</a>',1)
    head,body=t.split('<!--HEAD-END-->',1)
    doc=f'<!doctype html>\n<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{head}</head><body>{body}\n</body></html>'
    open(out,'w',encoding='utf-8').write(doc)
build('template.html',m['model'],w,'../tr.html','tr')
V={'irk':('Race / ethnicity',{1:'White (non-Hispanic)',2:'Black (non-Hispanic)',8:'Hispanic',4:'Asian',3:'American Indian / Alaska Native',5:'Native Hawaiian / Pacific Islander',7:'Multiracial',6:'Other'}),
'egitim':('Education',{1:'No schooling',2:'Grades 1–8',3:'Some high school',4:'High school graduate',5:'Some college / technical school',6:'College graduate'}),
'gelir8':('Annual household income',{1:'< $10k',2:'$10–15k',3:'$15–20k',4:'$20–25k',5:'$25–35k',6:'$35–50k',7:'$50–75k',8:'$75k or more'}),
'medeni':('Marital status',{1:'Married',2:'Divorced',3:'Widowed',4:'Separated',5:'Never married',6:'Unmarried couple'}),
'istihdam':('Employment',{1:'Employed for wages',2:'Self-employed',3:'Out of work (1 year+)',4:'Out of work (<1 year)',5:'Homemaker',6:'Student',7:'Retired',8:'Unable to work'}),
'ev_sahibi':('Housing',{1:'Own',2:'Rent',3:'Other arrangement'}),'kirsal':('Residence',{0:'Urban',1:'Rural'}),'dil_ispanyolca':('Interview language',{0:'English',1:'Spanish'}),
'sigortasiz':('Health insurance',{0:'Yes',1:'No'}),'doktor':('Personal doctor',{1:'Yes, one',2:'Yes, more than one',3:'No'}),'maliyet_engeli':('Could not see a doctor because of cost (past 12 months)',{1:'Yes',2:'No'}),
'checkup':('Last routine check-up',{1:'Within the past year',2:'1–2 years ago',3:'2–5 years ago',4:'5 or more years ago',8:'Never'}),'genel_saglik':('Self-rated health',{1:'Excellent',2:'Very good',3:'Good',4:'Fair',5:'Poor'}),
'sigara':('Smoking',{1:'Smokes every day',2:'Smokes some days',3:'Former smoker',4:'Never smoked'}),'grip_asisi':('Flu shot in the past 12 months',{1:'Yes',2:'No'}),'hiv_testi':('Ever tested for HIV',{1:'Yes',2:'No'}),
'dis_hekimi':('Last dental visit',{1:'Within the past year',2:'1–2 years ago',3:'2–5 years ago',4:'5 or more years ago',8:'Never'})}
me=copy.deepcopy(m['model'])
for v,(l,lv) in V.items():
    assert [c for c,_ in me['vars'][v]['levels']]==list(lv), v
    me['vars'][v]=dict(label=l,levels=[[c,t] for c,t in lv.items()])
T={'Yaş grubu':'Age group','Sağlık sigortası':'Health insurance','Düzenli doktor':'Personal doctor','Maliyet engeli':'Cost barrier','Son check-up':'Last check-up','Eğitim':'Education','Hane geliri':'Household income','Irk / etnisite':'Race / ethnicity','Medeni durum':'Marital status','Yerleşim':'Residence','Sigara':'Smoking','Görüşme dili':'Interview language',
'Sigortalı':'Insured','Sigortasız':'Uninsured','Doktoru var':'Has a personal doctor','Düzenli doktoru yok':'No personal doctor','Maliyet engeli var':'Cost barrier','Maliyet engeli yok':'No cost barrier','Son 1 yıl içinde':'Within the past year','1–2 yıl önce':'1–2 years ago','2–5 yıl önce':'2–5 years ago','5 yıldan eski':'5 or more years ago','Hiç':'Never',
'Lise altı':'Less than high school','Lise mezunu':'High school graduate','Üniversite terk / ön lisans':'Some college','Üniversite mezunu':'College graduate','< 15 bin $':'< $15k','15–25 bin $':'$15–25k','25–35 bin $':'$25–35k','35–50 bin $':'$35–50k','50–75 bin $':'$50–75k','75 bin $ ve üzeri':'$75k or more',
'Beyaz':'White','Siyah':'Black','Hispanik':'Hispanic','Asyalı':'Asian','Amerikan Yerlisi':'American Indian / Alaska Native','Pasifik Adalı':'Pacific Islander','Çok ırklı':'Multiracial','Diğer':'Other','Evli':'Married','Boşanmış':'Divorced','Dul':'Widowed','Ayrı yaşıyor':'Separated','Hiç evlenmemiş':'Never married','Partneriyle yaşıyor':'Unmarried couple',
'Kentsel':'Urban','Kırsal':'Rural','Her gün içiyor':'Smokes every day','Bazı günler içiyor':'Smokes some days','Bırakmış':'Former smoker','Hiç içmemiş':'Never smoked','İngilizce':'English','İspanyolca':'Spanish',
'Lojistik regresyon':'Logistic regression','Karar ağacı':'Decision tree','Yapay sinir ağı (MLP)':'Neural network (MLP)','LightGBM (tüm 37 değişken + eyalet)':'LightGBM (all 37 variables + state)',
'21–29 yaş':'Ages 21–29','30–49 yaş':'Ages 30–49','50–65 yaş':'Ages 50–65','İngilizce görüşme':'English interview','İspanyolca görüşme':'Spanish interview',
'BRFSS 2020, tüm görüşmeler':'BRFSS 2020, all interviews','Kadınlar':'Women','21–65 yaş':'Ages 21–65','Histerektomi geçirmemiş':'No hysterectomy','Tarama bilgisi tam (analitik örneklem)':'Complete screening items (analytic sample)','Güncel':'Up to date','Geri kalmış':'Overdue','Hiç taranmamış':'Never screened'}
T.update({'21–25 yaş':'Ages 21–25','26–39 yaş':'Ages 26–39','40–54 yaş':'Ages 40–54','55–65 yaş':'Ages 55–65','Boşanmış, dul ya da ayrı':'Divorced, widowed or separated','Çocuklu hane':'Children in household','Çalışıyor':'Employed','Öğrenci':'Student','İşsiz':'Out of work','Ev hanımı':'Homemaker','Çalışamıyor':'Unable to work',
'Lise veya altı':'High school or less','Gelir < 35 bin $':'Income < $35k','Gelir ≥ 75 bin $':'Income ≥ $75k','Kiracı':'Renter','İspanyolca görüşme':'Spanish interview','Maliyet yüzünden doktora gidememiş':'Could not see a doctor because of cost','Son check-up 2 yıldan eski ya da hiç':'Last check-up more than 2 years ago or never',
'Diş hekimine 2+ yıldır gitmemiş':'No dental visit in 2+ years','Grip aşısı olmamış':'No flu shot','Hiç HIV testi yaptırmamış':'Never tested for HIV','Halen sigara içiyor':'Current smoker','Sağlığı orta ya da kötü':'Fair or poor health',
'Genç ve bekâr kadınlar, çoğu öğrenci':'Young single women, many of them students','Çalışan, evli ya da orta yaşlı kadınlar':'Working, married or mid-life women','Düşük gelirli, sigortasız Hispanik anneler':'Low-income, uninsured Hispanic mothers',
'Varlıklı, evli, beyaz orta yaş kadınlar':'Affluent, married, White mid-life women','Düşük gelirli, sigara içen, bakımdan kopmuş kadınlar':'Low-income smokers disconnected from care','İspanyolca konuşan, sigortasız Hispanik kadınlar':'Spanish-speaking, uninsured Hispanic women','Eğitimli, çalışan Asyalı kadınlar':'Educated, working Asian women',
'Varlıklı, evli, beyaz kadınlar':'Affluent, married, White women','Genç, bekâr, düşük gelirli kiracılar':'Young, single, low-income renters','İspanyolca konuşan Hispanik kadınlar':'Spanish-speaking Hispanic women','Eğitimli Asyalı kadınlar':'Educated Asian women'})
T.update({'Eyalet':'State','Kronik hastalıklar ve engellilik (19 soru)':'Chronic conditions and disability (19 items)','Yaşam tarzı: egzersiz, uyku, BMI, alkol, emniyet kemeri (6 soru)':'Lifestyle: exercise, sleep, BMI, alcohol, seat belt (6 items)',
'HIV risk davranışı ve gebelik (2 soru)':'HIV risk behaviour and pregnancy (2 items)','Hane yapısı: hanedeki yetişkin sayısı':'Household: number of adults','Tüm çekirdek ek sorular birlikte (29 soru)':'All extra core items together (29 items)','Mamografi öyküsü (40–65 yaş)':'Mammography history (ages 40–65)',
'HPV aşısı modülü (18–49 yaş, modülü soran eyaletler)':'HPV vaccination module (ages 18–49, states fielding it)','Cinsel yönelim ve cinsiyet kimliği modülü':'Sexual orientation and gender identity module','Çocukluk çağı olumsuz deneyimleri modülü (11 soru + skor)':'Adverse childhood experiences module (11 items + score)',
'Sigorta türü modülü':'Insurance type module','E-sigara, esrar, tetanos aşısı, bakım verme (modüller)':'E-cigarettes, marijuana, tetanus shot, caregiving (modules)',
'Son check-up zamanı: kadınların %20’sinde bir basamak kayık':'Time since check-up: off by one category for 20% of women','Gelir: kadınların %30’unda bir basamak kayık':'Income: off by one bracket for 30% of women','HIV testi: kadınların %10’unda yanıt ters':'HIV test: answer flipped for 10% of women','Yaş: ±3 yıl rastgele hata':'Age: random error of ±3 years',
'Hepsi birlikte (check-up, diş hekimi, gelir, HIV testi, yaş)':'All together (check-up, dental visit, income, HIV test, age)','Daha genç nüfus (30 yaş altı 3 kat ağırlık)':'Younger population (under 30 weighted ×3)','Daha çok sigortasız (3 kat)':'More uninsured (×3)','Daha çok Hispanik (3 kat)':'More Hispanic (×3)',
'Daha düşük gelirli (25 bin $ altı 3 kat)':'Lower income (under $25k ×3)','Daha kırsal (3 kat)':'More rural (×3)','Düzenli doktoru olmayanlar (3 kat)':'No personal doctor (×3)','Yaş':'Age','Hanedeki çocuk sayısı':'Children in household','Her sorunun %10 kadarı yanıtsız':'10% of every item unanswered','Her sorunun %30 kadarı yanıtsız':'30% of every item unanswered','Her sorunun %50 kadarı yanıtsız':'50% of every item unanswered'})
for v,(l,lv) in V.items(): T[m['model']['vars'][v]['label']]=l
tr=lambda x:('Exploratory profile '+x.rsplit(' ',1)[-1]) if isinstance(x,str) and x.startswith('Keşifsel profil ') else T.get(x,x)
we=copy.deepcopy(w)
for row in we.get('fairness',[]): row['label']=row.pop('label_en')
for q in we['blocks']: q['blok']=tr(q['blok'])
for k in ('hatali','kayma','eksik'):
    for r in we['stress'][k]: r['senaryo']=tr(r['senaryo'])
for r in we['stress']['soru_yok']: r['soru']=tr(r['soru'])
for r in we['stress']['kisa_form']: r['eklenen']=tr(r['eklenen'])
for d in we['desc']:
    d['label']=tr(d['label'])
    for r in d['rows']: r[0]=tr(r[0])
for c in we['cv']: c['model']=tr(c['model'])
for g in we['sub']: g['grup']=tr(g['grup'])
for R in we['rev'].values():
    for r in R['comp']: r[0]=tr(r[0])
    for c in R['clusters']:
        c['name']=tr(c['name'])
        for t in c['top']+c['low']: t[0]=tr(t[0])
we['flow']=[[tr(a),b] for a,b in we['flow']]; we['classes']=[[tr(a),b,c] for a,b,c in we['classes']]
build('template_en.html',me,we,'../index.html','en')
left=set(re.findall(r"[^\"'`<>,:\[\]{}]*[ğşıİöüçĞŞÖÜÇ][^\"'`<>,:\[\]{}]*",open('../index.html',encoding='utf-8').read()))
left=[x for x in left if '/*' not in x and x.strip()!='Türkçe']
if left: raise ValueError('Untranslated Turkish text remains in index.html: '+repr(left))
os.makedirs('../out',exist_ok=True)
open('../out/tests.json','w').write(js(m.get('tests',[])))
