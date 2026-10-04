#!/usr/bin/env bash
# Full reproduction. Expects LLCP2020.XPT, LLCP2022.XPT and LLCP2024.XPT in the repository root (LLCP2021/2023 only for the supplementary multi-year analysis).
# Python 3.12; pip install -r requirements.txt. Total run time is roughly 4–7 hours on two cores (steps 07, 10 and 13 dominate).
set -e; export PYTHONUTF8=1; mkdir -p out
python code/00_extract.py 2020; python code/00_extract.py 2024
python code/00b_extract_screening_items.py 2022; python code/00b_extract_screening_items.py 2024
python code/01_define_classes.py
python code/02_model_selection_cv.py
python code/03_final_model_validation_export.py
python code/04_threshold_table.py
python code/05_class_composition.py
python code/06_extra_question_blocks.py
python code/07_stress_tests.py
python code/08_missing_answers.py
python code/09_two_stage_subgroups_dca.py
python code/10_algorithm_search.py
python code/11_tuned_models_confusion_shap.py
python code/12_question_wording.py
python code/13_design_bootstrap_grouped_cv_conformal.py 1000
python code/14_web_confidence_intervals.py
python code/check_outputs.py
(cd web_src && python en.py && python build.py)
python code/check_web.py --release
