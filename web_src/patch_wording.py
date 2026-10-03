# One-off patch (pre-submission): "external validation" → "temporal evaluation under outcome-measurement change";
# 200 iid bootstrap → 1,000 design-based resamples; softer causal language about the 2022/2024 instrument change.
import json
REP=[
("Geliştirme: BRFSS 2020 (98.847 kadın) · Dış doğrulama: BRFSS 2024 (104.650 kadın) · 21–65 yaş",
 "Geliştirme: BRFSS 2020 (98.847 kadın) · Zamansal değerlendirme, ölçüm değişikliği altında: BRFSS 2024 (104.650 kadın) · 21–65 yaş",
 "Development: BRFSS 2020 (98,847 women) · External validation: BRFSS 2024 (104,650 women) · ages 21–65",
 "Development: BRFSS 2020 (98,847 women) · Temporal evaluation under outcome-measurement change: BRFSS 2024 (104,650 women) · ages 21–65"),
("2020'de 45–55 yaşında olup %3'ü hiç taranmamış olan kuşağın dört yıl sonra %11'inin hiç taranmamış görünmesi mümkün değildir; fark ölçümden kaynaklanıyor. Bu nedenle model 2020 verisiyle geliştirildi.",
 "2020'de 45–55 yaşında olup %3,9'u hiç taranmamış görünen kuşak, 2022'de (önsözsüz soru) %25,4 ve 2024'te (tanımlayıcı önsözlü soru) %10,8 ile görünmektedir; farkın büyük ölçüde soru ifadesiyle ilişkili olduğu düşünülmektedir. Model bu nedenle 2020 verisiyle geliştirildi; 2024 sonuçları ölçüm değişikliği altındaki performansı tanımlar.",
 "A cohort aged 45–55 in 2020, of whom 3% had never been screened, cannot show 11% never screened four years later; the gap is a measurement artefact. The model was therefore developed on 2020 data.",
 "A cohort aged 45–55 in 2020, of whom 3.9% appeared never screened, appears at 25.4% in 2022 (item without preamble) and 10.8% in 2024 (item with a definitional preamble); the difference is thought to be associated largely with the wording of the item. The model was therefore developed on 2020 data, and the 2024 results describe performance under the changed measurement."),
("<h2>İç ve dış doğrulama</h2>","<h2>İç doğrulama ve 2024 zamansal değerlendirmesi</h2>",
 "<h2>Internal and external validation</h2>","<h2>Internal validation and temporal evaluation in 2024</h2>"),
("İç doğrulama: 2020 verisinin ayrılmış %20'si (<span id=\"ntest\"></span> kadın). Dış doğrulama: 2020'de eğitilen modelin hiç görmediği BRFSS 2024 verisi (<span id=\"n24\"></span> kadın). Parantez içi %95 güven aralığıdır (200 bootstrap).",
 "İç doğrulama: 2020 verisinin ayrılmış %20'si (<span id=\"ntest\"></span> kadın). Zamansal değerlendirme: aynı hiperparametrelerle tam 2020 örneklemine yeniden uydurulan modelin, farklı bir tarama sorusu kullanan BRFSS 2024 verisine (<span id=\"n24\"></span> kadın) uygulanması. 2024 satırları ölçüm değişikliği altındaki performansı tanımlar ve dış doğrulama olarak okunmamalıdır. Parantez içi %95 güven aralığıdır (örnekleme tabakaları içinde 1.000 tasarım temelli yeniden örneklem).",
 "Internal validation: a held-out 20% of the 2020 data (<span id=\"ntest\"></span> women). External validation: BRFSS 2024, never seen by the model trained on 2020 (<span id=\"n24\"></span> women). Brackets give 95% confidence intervals (200 bootstrap samples).",
 "Internal validation: a held-out 20% of the 2020 data (<span id=\"ntest\"></span> women). Temporal evaluation: the model refitted to the full 2020 sample with the same hyperparameters, applied to BRFSS 2024 (<span id=\"n24\"></span> women), which used a different screening item. The 2024 columns describe performance under a change in outcome measurement and should not be read as an external validation. Brackets give 95% confidence intervals (1,000 design-based resamples within sampling strata)."),
("Dördüncü panel 2024 dış doğrulamasında, taranmış kadınlar içinde geri kalma olasılığını gösterir.",
 "Dördüncü panel 2024 zamansal değerlendirmesinde, taranmış kadınlar içinde geri kalma olasılığını gösterir.",
 "The fourth shows the probability of being overdue among ever-screened women in the 2024 external validation.",
 "The fourth shows the probability of being overdue among ever-screened women in the 2024 temporal evaluation."),
("Model 2020 dosyasıyla (401.958 görüşme) geliştirildi, 2024 dosyasıyla (457.670 görüşme) dışarıdan sınandı. 2021–2023 dosyaları kullanılmadı: tek yıllarda tarama soruları yalnızca birkaç eyalette sorulmuş, 2022'de ise \"hiç taranmamış\" oranı %38 gibi gerçekçi olmayan bir düzeye çıkmıştır; CDC de o yıl için serviks taraması hesaplanmış değişkeni yayımlamamıştır.",
 "Model 2020 dosyasıyla (401.958 görüşme) geliştirildi ve 2024 dosyasında (457.670 görüşme), değişen tarama sorusu altında değerlendirildi. 2021–2023 dosyaları modelleme için kullanılmadı: tek yıllarda tarama soruları yalnızca birkaç eyalette sorulmuş, 2022'de ise tanımlayıcı önsözü olmayan yeni soruyla \"hiç taranmamış\" oranı %37'ye yükselmiştir; CDC de o yıl için serviks taraması hesaplanmış değişkeni yayımlamamıştır.",
 "The model was developed on the 2020 file (401,958 interviews) and externally tested on the 2024 file (457,670 interviews). The 2021–2023 files were not used: in odd years the screening items were fielded in only a few states, and in 2022 the never-screened share reached an implausible 38%; CDC released no calculated cervical screening variable for that year.",
 "The model was developed on the 2020 file (401,958 interviews) and evaluated on the 2024 file (457,670 interviews) under a changed screening item. The 2021–2023 files were not used for modelling: in odd years the screening items were fielded in only a few states, and in 2022, under the new item without a definitional preamble, the never-screened share rose to 37%; CDC released no calculated cervical screening variable for that year."),
("Aynı kuşak izlendiğinde bu artış gerçek olamaz; kadınların bir bölümü yeni ifadeyi tanımamaktadır. \"Geri kalmış\" oranları ise iki yılda da aynıdır. Bu yüzden etiketleri güvenilir olan 2020 verisi geliştirme kümesi seçildi.",
 "Aynı doğum kuşağı izlendiğinde de fark görülür (2020'de %3,9, 2022'de %25,4, 2024'te %10,8) ve 2022 fazlası tüm eyaletlerde vardır; farkın büyük ölçüde soru ifadesiyle ilişkili olduğu düşünülmektedir. \"Geri kalmış\" oranları ise iki yılda da birbirine yakındır. Bu yüzden Pap testi maddelerini kullanan 2020 verisi geliştirme kümesi seçildi.",
 "Following the same cohorts, such an increase cannot be real; some women do not recognise the new wording. Overdue rates, by contrast, are the same in both years. The 2020 data, with reliable labels, was therefore chosen for development.",
 "The difference is also seen when the same birth cohort is followed (3.9% in 2020, 25.4% in 2022, 10.8% in 2024), and the 2022 excess is present in every state; it is thought to be associated largely with the wording of the item. Overdue rates, by contrast, are similar in both years. The 2020 data, which used the Pap-test items, was therefore chosen for development."),
("['Ölçüt','İç doğrulama (2020)','Dış doğrulama (2024)']","['Ölçüt','İç doğrulama (2020)','Zamansal değerlendirme (2024)']",
 "['Metric','Internal validation (2020)','External validation (2024)']","['Metric','Internal validation (2020)','Temporal evaluation (2024)']"),
]
t=open('template.html',encoding='utf-8').read(); P=json.load(open('en_pairs.json',encoding='utf-8'))
for tr0,tr1,en0,en1 in REP:
    assert tr0 in t, tr0[:50]; t=t.replace(tr0,tr1)
    hit=[i for i,(a,b) in enumerate(P) if tr0 in a and en0 in b]; assert hit, en0[:50]; a,b=P[hit[0]]; P[hit[0]]=[a.replace(tr0,tr1),b.replace(en0,en1)]
open('template.html','w',encoding='utf-8').write(t); json.dump(P,open('en_pairs.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
print('patched',len(REP))
