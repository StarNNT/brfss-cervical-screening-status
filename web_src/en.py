# Türkçe şablondan İngilizce şablon üretir (en_pairs.json: [türkçe, ingilizce] çiftleri)
import json,re,sys
s=open('template.html',encoding='utf-8').read(); P=json.load(open('en_pairs.json',encoding='utf-8'))
for a,b in sorted(P,key=lambda p:-len(p[0])):
    s=s.replace(a,b)
open('template_en.html','w',encoding='utf-8').write(s)
left=[]
for i,l in enumerate(s.split('\n'),1):
    for m in re.finditer(r"[^'\"`<>]*[ğşıİöüçĞŞÖÜÇ][^'\"`<>]*",l):
        if '/* --' not in m.group(0): left.append((i,m.group(0)[:110]))
if left:
    for i,x in left: print('KALAN',i,x)
    sys.exit('English template still contains untranslated Turkish text')
