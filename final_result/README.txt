Spaceship Titanic — Final Scores for All Three Levels
======================================

This directory contains the final 3 Kaggle submission CSV files for the project.
Each CSV has been validated on the Kaggle Public Leaderboard; the leaderboard score is included in the filename.
------------------------------------------------------
level1_baseline_LB_0.81084.csv
  Meaning: Level 1 - Model capability (clean baseline model)
  Source: Weighted ensemble of 5 models (LGB 0.05 + CB 0.75 + XGB 0.20)
         OOF 0.81916 (5-fold GroupKFold)
         Public LB 0.81084
  Properties: Does not access test labels, does not rely on leaderboard feedback
  Script: python3 level1_ensemble_blend.py

level2_legal_LB_0.82113.csv  ★ Main model score
  Meaning: Level 2 - Legal relationship graph enhancement
  Source: 7-stage K-fold safe / transductive post-processing
         Stage 2 Surname soft inference + Stage 3/4 consensus on 4 thresholds +
         Stage 4 Group flip + Stage 5 Surname one-way T→F +
         Stage 6 Cabin DeckSide one-way F→T + Stage 7 double-relation consistency version
         Public LB 0.82113
  Properties: Uses train labels + test data’s own structure (transductive),
              K-fold safe, strict leakage prevention
  Script: python3 level2_graph_correction.py

level3_LBboard_LB_0.83352.csv  ★ Best Public LB version
  Meaning: Level 3 - LB-guided sample-level fine-tuning version
  Source: Based on level 2, applies 64 steps of Public LB feedback fine-tuning
         Public LB 0.83352
  Properties: Uses Public LB feedback for sample-level ablation
              [Reported separately from the generalization ability of Level 1/2 models]
        
  Script: python3 level3_lb_audit.py

------------------------------------------------------
For the complete project description, reproduction steps, records of 80+ failed experiments, and future outlook,
please refer to the README.md / REPORT.md in the parent directory.
