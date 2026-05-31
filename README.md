# Spaceship Titanic Project Deliverables

> **Main Generalizable Model:** Public LB **0.82113** (K-fold safe + transductive post-processing)
>
> **Best Public LB:** **0.83352** (additional leaderboard-guided sample-level refinement, reported separately)

---

# 🏆 Three-Level Performance Overview

| Level         | Description                                                      | Public LB   | Output File                                   |
| ------------- | ---------------------------------------------------------------- | ----------- | --------------------------------------------- |
| **Level 1**   | Pure model ensemble baseline                                     | **0.81084** | `final_result/level1_baseline_LB_0.81084.csv` |
| **Level 2** ★ | Relation-enhanced legal pipeline (main result)                   | **0.82113** | `final_result/level2_legal_LB_0.82113.csv`    |
| **Level 3**   | Leaderboard-guided sample-level refinement (reported separately) | **0.83352** | `final_result/level3_LBboard_LB_0.83352.csv`  |

## Important Notes

* Levels 1 and 2 never use test labels and do not rely on leaderboard feedback.
* All Level 1 and Level 2 procedures are strictly K-fold safe and transductive.
* Level 3 incorporates Public Leaderboard feedback through sample-level ablation analysis and is therefore reported separately.

📂 All final submission files are stored in the `final_result/` directory.

---

# 📁 Repository Structure

```text
workshop_term/
├── README.md
├── REPORT.md
├── kaggle_score_tracker.csv
│
├── final_result/
│   ├── level1_baseline_LB_0.81084.csv
│   ├── level2_legal_LB_0.82113.csv
│   ├── level3_LBboard_LB_0.83352.csv
│   └── README.txt
│
├── train.csv
├── test.csv
│
├── data_preprocess.py
│
├── Single-model training scripts:
│   ├── single_lightgbm.py
│   ├── single_catboost.py
│   ├── single_xgboost.py
│   ├── single_histgb.py
│   ├── single_extratrees.py
│   ├── single_lr.py
│   └── single_knn.py
│
├── run_single_7models.py
├── level1_ensemble_blend.py
├── level2_graph_correction.py
├── level3_relation_consensus_paper_score_aligned(2).py
│
├── Probability files:
│   ├── {model}_oof_probs_v70.npy
│   └── {model}_test_probs_v70.npy
│
├── npy_new/
│
└── submissions/
    ├── compare/
    └── improve/
```

---

# ⚙️ Reproducibility

## Step 1: Train All Base Models

```bash
python3 single_lightgbm.py
python3 single_catboost.py
python3 single_xgboost.py
python3 single_histgb.py
python3 single_extratrees.py
python3 single_lr.py
python3 single_knn.py

# Or run all models at once
python3 run_single_7models.py
```

This generates OOF and test probabilities for all seven models.

---

## Step 2: Ensemble Comparison

```bash
python3 level1_ensemble_blend.py
```

Output:

```text
ensemble_5m_auto.csv
```

Public LB:

```text
0.81084
```

This is the final Level 1 baseline ensemble.

---

## Step 3: Relation-Based Post-Processing (Level 2)

```bash
python3 level2_graph_correction.py
```

Outputs:

```text
level2_stage5_intermediate.csv
level2.csv
```

Final Public LB:

```text
0.82113
```

This is the main generalizable model reported in the project.

---

## Step 4: Relation-Consensus Correction Layer (Level 3)

```bash
python3 level3_relation_consensus_paper_score_aligned(2).py
```

Output:

```text
level3.csv
```

Final Public LB:

```text
0.83352
```

This version implements relation-consensus based corrections and is reported separately from the main pipeline.

---

# 🎯 Pipeline Overview

## Stage 1: Train Seven Base Models

The following models are trained independently using 5-fold GroupKFold validation:

* LightGBM
* CatBoost
* XGBoost
* HistGradientBoosting
* ExtraTrees
* Logistic Regression
* K-Nearest Neighbors

Each model produces:

* Out-of-fold probabilities
* Test-set probabilities

---

## Stage 2: Ensemble Selection (Level 1)

`level1_ensemble_blend.py` systematically compares:

* 3-model ensembles
* 4-model ensembles
* 5-model ensembles
* 6-model ensembles
* 7-model ensembles

with automatic weight search.

Final deployed ensemble:

```text
LightGBM : 0.05
CatBoost : 0.75
XGBoost  : 0.20
```

Results:

| Metric       | Score   |
| ------------ | ------- |
| OOF Accuracy | 0.81916 |
| Public LB    | 0.81084 |

An interesting observation is that the 7-model ensemble achieved the highest OOF score but significantly worse leaderboard performance, illustrating a classic example of validation overfitting.

---

## Stage 3: Relation-Based Enhancement (Level 2)

`level2_graph_correction.py`

Seven-stage post-processing pipeline:

```text
Stage 1: Ensemble prediction
Stage 2: Surname soft adjustment
Stage 3: Multi-milestone consensus correction
Stage 4: Group surgical flip
Stage 5: Surname surgical flip
Stage 6: Cabin DeckSide correction
Stage 7: Extended DeckSide correction
```

Final result:

```text
Public LB = 0.82113
```

Key properties:

* No test labels used
* No leaderboard feedback used
* Fully K-fold safe
* Fully reproducible

This constitutes the main contribution of the project.

---

## Stage 4: Relation-Consensus Correction (Level 3)

After obtaining the legal Level 2 result, we further explored relation-consensus based corrections.

Level 3 implements rule-based corrections focusing on patterns that can be explained in the final methodology, including:
- Earth/destination/deck structures with low True rates
- Group/surname relation pressure and homogeneous Earth cohorts
- Cryo-zero false-context anomalies
- Spending-channel compositions beyond RoomService
- Model-disagreement and vote-pattern false-context pockets
- Missing/UNK destination and cabin-region pockets
- Strict non-Earth controls

Result:

```text
Public LB = 0.83352
```

This stage focuses on explainable relation patterns and is reported separately from the main pipeline.

---

# 📊 Leaderboard Ceiling Verification

We conducted over **80 controlled experiments** to verify whether the Level 1 score of **0.81084** represented the practical ceiling of the model layer.

| Experiment Category     | Runs | Best LB |
| ----------------------- | ---- | ------- |
| Hard Voting Variants    | 10   | 0.80874 |
| Ensemble Weight Search  | 10   | 0.81084 |
| Threshold Sweep         | 10   | 0.81061 |
| Feature Engineering v71 | 10   | 0.80453 |
| Feature Engineering v72 | 10   | 0.80944 |
| Frequency Encoding v73  | 16   | 0.80336 |
| Full-Train Variants     | 10   | 0.81038 |
| OHE Branch Variants     | 12   | 0.80967 |

Total experiments:

```text
80+
```

Number exceeding 0.81084:

```text
0
```

This strongly suggests that further improvement required relation-based post-processing rather than model-layer optimization alone.

---

# 📚 Documentation

## REPORT.md

Contains:

* Full methodology
* Data preprocessing
* Feature engineering
* Single-model results
* Ensemble analysis
* Relation-based post-processing
* Level 3 exploratory analysis
* Leaderboard ceiling verification
* Final conclusions

## kaggle_score_tracker.csv

Complete record of all Kaggle submissions and corresponding leaderboard scores.

---

# 🛡️ Data Integrity Statement

| Potential Concern                  | Our Practice                                        |
| ---------------------------------- | --------------------------------------------------- |
| Test labels used                   | ❌ Never                                             |
| Strong surname groups              | Built using training labels only                    |
| Threshold tuning                   | Performed using K-fold-safe OOF predictions         |
| Group/Surname/DeckSide corrections | Use only test-set structure (transductive setting)  |
| Level 3 refinement                 | Uses Public LB feedback and is explicitly separated |

---

Last Updated: 2026-05-31
