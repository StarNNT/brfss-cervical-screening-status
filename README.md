# Cervical screening outreach prioritization

This repository contains the analysis code and browser-based implementation for estimating survey-derived **up to date**, **overdue** and **never screened** classes from sociodemographic, access-to-care and health-behaviour characteristics in the US CDC Behavioral Risk Factor Surveillance System (BRFSS). The estimates are intended for research on voluntary population outreach, not for cancer-risk assessment or individual clinical decisions.

**Web tool:** [English](index.html) · [Türkçe](tr.html).

The tool is a single static page. The model runs in the browser; nothing a user enters leaves their computer. The page loads no external scripts or fonts. It must not be used to deny, delay or ration care.

> **Current analysis status (version 1.1.0, 2026-09-29):** the revised label definition requires an explicit “No” to both Pap and HPV items for the never-screened class; unknown and refused HPV responses are not treated as “No”. The source-data audit gives n = 98,972 (weighted: 80.1% up to date, 12.2% overdue, 7.7% never screened). The coefficients in the HTML files predate this revision. Rerun the complete analysis before publication or decision-impact evaluation.

## Study in brief

| | |
|---|---|
| Development data in the stored model version | BRFSS 2020 — participants classified by BRFSS as women, aged 21–65, reporting no hysterectomy (n = 98,847 under the previous label rule) |
| Temporal evaluation | BRFSS 2024, same eligibility (n = 104,650), under a changed screening instrument |
| Outcome (2020, weighted) | Up to date 80.3% · overdue 11.6% · never screened 8.1% |
| Final model | Multinomial logistic regression, 19 inputs, piecewise-linear age, survey-weighted |
| Internal validation (held-out 20%; stored model version) | AUC never screened 0.847 (design-based 95% CI 0.824–0.868) · overdue 0.780 (0.756–0.803) · up to date 0.767 (0.749–0.784) |
| 2024 evaluation (stored model version) | AUC never screened 0.790 (0.781–0.798) · overdue 0.724 (0.714–0.734) · up to date 0.742 (0.735–0.749) |
| Algorithm comparison | Tuned LightGBM/XGBoost/CatBoost and ensembles: within 0.003 macro AUC of logistic regression on held-out 2020 (CI includes zero for never-screened AUC); +0.011 never-screened AUC in 2024 |
| Uncertainty | Design-based bootstrap (1,000 within-stratum resamples) for every metric; state-grouped 10-fold CV |
| Profiles | Weighted decision trees (depth 4) with design-based 95% confidence intervals |
| Extra questions | Of twelve blocks of additional real BRFSS items, only mammography history adds materially (mean AUC 0.76 → 0.82 in women aged 40–65) |
| Sensitivity analyses | Simulated reporting error, reweighting and missing-answer scenarios; the first 8 questions reach mean AUC 0.790 vs. 0.798 for all 19 in the stored model version |

Seven algorithm families (logistic regression, decision tree, random forest, XGBoost, LightGBM, CatBoost, neural network) were compared with 5-fold cross-validation, the three boosting implementations also after a random hyperparameter search, and two ensembles were formed. Logistic regression was within 0.005 macro AUC of the tuned boosting models and was chosen for deployment because it is transparent and runs exactly in the browser; the tuned models and ensembles are reported alongside.

### Intended use and exclusions

The output is an outreach-prioritization estimate, not a diagnosis of screening status or cancer risk. The development population was the US BRFSS research sample described above. The tool does not assess current clinical eligibility and is outside scope for people with prior CIN2/3 or cervical cancer, immunosuppression, in-utero DES exposure, symptoms, abnormal results, or other special follow-up requirements. Race/ethnicity and language are treated as markers of structural access inequities, not biological causes. Local prospective evaluation must examine calibration, sensitivity, false-positive rate, workload, subgroup disparities and actual outreach benefit before implementation.

The 2020 outcome is an **USPSTF 2018 interval proxy**: up to date means a Pap test within 3 years **or**, at age 30 or older, an HPV test within 5 years; never screened requires explicit No responses to both ever-Pap and ever-HPV items; overdue means a known prior Pap or HPV test that does not meet the up-to-date rule. Ambiguous histories are excluded rather than assumed to be never screened. These are survey-derived labels, not clinical-record verification.

### Why the model is developed on 2020, and why 2024 is a temporal evaluation rather than an external validation

The BRFSS used three instruments for the "ever screened" item between 2020 and 2024: "Have you ever had a Pap test?" (2020), the bare "Have you ever had a cervical cancer screening test?" (2021–2023; core in 2022, optional module in 2021 and 2023) and, in 2024, the same item preceded by "There are two different kinds of tests to check for cervical cancer. One is a Pap smear or Pap test and the other is the HPV or Human Papillomavirus test." The reported never-screened share among women aged 21–65 without hysterectomy was 9.0%, 37.4% and 17.6% under the three instruments, the share of "don't know"/refused answers 0.4%, 6.6% and 2.2%, and the same birth cohort reported 3.9%, 25.4% and 10.8%. The 2022 excess is present in every state (median ratio to 2024 of 2.15). Overdue rates, from items whose wording did not change, are nearly identical across years. The 2020 labels are therefore used for development, and the 2024 file is a temporal evaluation under outcome-measurement change: the overdue class is the cleaner test, and the never-screened results for 2024 describe the model under a changed outcome definition. `code/12_question_wording.py` reproduces these figures.

## Reproducing the analysis

1. Download `LLCP2020XPT.zip`, `LLCP2022XPT.zip` and `LLCP2024XPT.zip` from the [BRFSS annual data pages](https://www.cdc.gov/brfss/annual_data/annual_data.htm) and unzip the `.XPT` files into the repository root. Raw files are not redistributed here.
2. `pip install -r requirements.txt`
3. Run from the repository root, in order. If the extracted `brfss*_kadin.csv.gz` and `tabaka_*.csv` files are elsewhere, set `BRFSS_DATA_DIR` for scripts 01 and 03 (for example, `BRFSS_DATA_DIR=../analiz` in a POSIX shell):

| Script | Purpose |
|---|---|
| `code/00_extract.py 2020` and `... 2024` | Reads the XPT file in chunks, keeps women and the study variables; writes stratum sizes |
| `code/01_define_classes.py` | Eligibility criteria and the three-class outcome for both question formats |
| `code/harmon.py` | Predictor recoding harmonised across 2020 and 2024 (imported by the scripts below) |
| `code/02_model_selection_cv.py` | 5-fold cross-validated comparison with default settings (logistic regression, decision tree, random forest, XGBoost, LightGBM, MLP); CatBoost and the tuned configurations are added in script 10, making seven algorithm families in all |
| `code/03_final_model_validation_export.py` | Final logistic model (fitted to the development set for the held-out evaluation and refitted to the full 2020 sample for 2024 and for the tool), calibration, lift, subgroup performance, SHAP, profiles with design-based CIs, export for the web page |
| `code/04_threshold_table.py` | Sensitivity, specificity and precision at each predicted-probability threshold (feeds the outreach-capacity slider in the tool); run after script 03 |
| `code/05_class_composition.py` | Reverse view: composition of each screening class, over-represented characteristics and within-class types (weighted k-means); run after script 03 |
| `code/06_extra_question_blocks.py` | Incremental value of real BRFSS items not used by the tool (chronic conditions, lifestyle, state, mammography, HPV vaccination, ACEs, …): base vs. base + block on the same folds |
| `code/07_stress_tests.py` | Robustness of the final model: misreported answers, population shift, short-form question ordering (slow: about 30 minutes) |
| `code/08_missing_answers.py` | Missing answers: survey "unknown" category vs. population-average fill; writes the column means the tool uses for unknown fields; run after 07 |
| `code/09_two_stage_subgroups_dca.py` | Two-stage hierarchical model, subgroup performance and decision curves (point estimates); run after 03 |
| `code/10_algorithm_search.py` | Random hyperparameter search (LightGBM 24, XGBoost 10, CatBoost 6 configurations) and 5-fold CV of all algorithms; writes `out/search_best.json` (slow, about 1.5 h) |
| `code/11_tuned_models_confusion_shap.py` | Tuned models and ensembles: fitted to the 80% development subset for the held-out set and refitted to the full 2020 sample (same hyperparameters) for the 2024 set; confusion matrices under the pre-specified rules only (no threshold selection on evaluation data), SHAP and coefficient-based importance, odds ratios from the refitted logistic model; reads `out/search_best.json` |
| `code/12_question_wording.py` | Instrument analysis: never-screened and don't-know shares, age pattern, birth cohorts and state spread for 2020, 2022 and 2024 |
| `code/13_design_bootstrap_grouped_cv_conformal.py` | Design-based bootstrap (1,000 within-stratum resamples) for all reported metrics and paired differences; thresholds selected on out-of-fold development predictions; state-grouped 10-fold CV; unweighted finite-sample split conformal sets; fold SDs of the tuned models |
| `code/14_web_confidence_intervals.py` | Replaces the iid-bootstrap intervals that script 03 writes for the web page with the design-based intervals of script 13, so the page and the paper report the same intervals |
| `code/check_outputs.py` | Compares regenerated key results with fixed expected values from the paper (tolerances allow for platform-level numerical differences) |

### Training scheme (which model is evaluated where)

Hyperparameters were selected by 5-fold cross-validation inside the 80% development subset of 2020 (script 10). Every model is then fitted twice with those hyperparameters: once to the development subset, and this fit is evaluated on the 20% held-out set; and once to the full 2020 sample, and this refit is evaluated on 2024 and, for the logistic regression, exported to the web tool (`out/webmodel.json`). The 2024 predictions in `out/model_predictions.npz` and the deployed model therefore come from the same fits. Confidence intervals in the paper, the tables and the web tool come from the design-based bootstrap of script 13 (1,000 within-stratum resamples; multiplicities are accumulated with `np.add.at`, since fancy-indexed `+=` would drop repeated draws).

This repository is the private submission-stage working copy associated with the manuscript. It will be made public upon acceptance after the corrected pipeline has been rerun, the manuscript values have been reconciled, and the release check passes. `run_all.sh` runs every step in order (about 4 hours on two cores). Dependency versions are pinned in `requirements.txt` (Python 3.11).

4. `cd web_src && python en.py && python build.py` rebuilds `../index.html` and `../tr.html`; `python ../code/check_web.py --release` then verifies that the static pages use coefficients from the current analysis outputs.

`code/supplementary/` holds exploratory analyses that are not part of the final model: a three-cancer comparison (cervical, breast, colorectal) on 2024 and a pooled 2021–2024 analysis with temporal validation. They document the data-quality findings above and the planned extension to other screenings.

## Limitations

BRFSS is cross-sectional and self-reported, and screening history tends to be over-reported. The development data date from 2020, so post-pandemic changes may not be reflected. Discrimination for never-screened status is weak among women aged 50–65, where very few women have never been screened. The 2024 outcome instrument changed, so that analysis is a temporal evaluation under measurement change, not a true external validation. Individual probabilities have no confidence intervals and subgroup errors are unequal. Findings describe association, not causation. The tool supports public-health research and voluntary outreach planning only. Results apply to the US research population and cannot be transferred directly to other countries, clinical settings or special-risk groups.

## Authors

- Veysel Gider, Distance Education Application and Research Center, Batman University, Türkiye
- Haluk Damlacıoğlu, Hollings Cancer Center, Medical University of South Carolina, United States
- Cafer Budak, Department of Electrical and Electronics Engineering, Dicle University, Türkiye

## Data source

Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System Survey Data, 2020 and 2024.* Atlanta, Georgia: U.S. Department of Health and Human Services. BRFSS data are in the public domain.
