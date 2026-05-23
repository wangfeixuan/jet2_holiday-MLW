Spaceship Titanic — 三层最终成绩
======================================

本目录是项目最终交付的 3 个 Kaggle 提交 csv.
每个 csv 都已在 Kaggle Public LB 上验证, LB 数字写在文件名里.
------------------------------------------------------
level1_baseline_LB_0.81084.csv
  含义: 第 1 层 - 模型能力 (干净基模型)
  来源: 5 模型加权融合 (LGB 0.05 + CB 0.75 + XGB 0.20)
        OOF 0.81916 (5-fold GroupKFold)
        Public LB 0.81084
  性质: 不接触 test 标签, 不依赖 LB 反馈
  脚本: python3 level1_ensemble_blend.py

level2_legal_LB_0.82113.csv  ★ 主模型成绩
  含义: 第 2 层 - 合法关系图增强
  来源: 7 段 K-fold safe / transductive 后处理
        Stage 2 Surname 软推 + Stage 3 4 阈值共识 +
        Stage 4 Group flip + Stage 5 Surname 单向 T→F +
        Stage 6 Cabin DeckSide 单向 F→T + Stage 7 双关系一致版
        Public LB 0.82113
  性质: 用 train 标签 + test 自身结构 (transductive),
        K-fold safe 严格防泄漏
  脚本: python3 level2_graph_correction.py

level3_LBboard_LB_0.83352.csv  ★ Public LB 最优版本
  含义: 第 3 层 - LB-guided 样本级精修版
  来源: 在 level 2 基础上, 应用 64 步 Public LB 反馈精修
        Public LB 0.83352
  性质: 使用 Public LB 反馈做样本级 ablation
        【诚实声明: 与第 1/2 层模型泛化能力分开汇报】
        【Private LB 揭晓后存在掉分风险, 详见 REPORT.md 第 7.4 节】
  脚本: python3 level3_lb_audit.py

------------------------------------------------------
完整的项目说明、复现步骤、80+ 失败实验记录、未来展望
请看上一层目录的 README.md / REPORT.md.
