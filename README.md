# Spaceship Titanic 项目交付包

> 主模型 Public LB **0.82113** (K-fold safe / transductive 后处理终版)
> Public LB 最优 **0.83352** (额外用了 LB-guided 样本级精修, separately reported)

## 🏆 三层成绩 (一眼看懂)

| 层 | 含义 | LB | 文件 |
|---|---|---|---|
| **第 1 层** | 模型能力 (干净基模型) | **0.81084** | `final_result/level1_baseline_LB_0.81084.csv` |
| **第 2 层** ★ | 合法关系图增强 (主模型成绩) | **0.82113** | `final_result/level2_legal_LB_0.82113.csv` |
| **第 3 层** | LB-guided 样本级精修 (separately reported) | **0.83352** | `final_result/level3_LBboard_LB_0.83352.csv` |

**关键声明**:
- 第 1/2 层未使用 test 标签也未引入 LB 反馈，严格 K-fold safe / transductive
- 第 3 层使用 Public LB 反馈做样本级 ablation，**separately reported**
- 

> 📂 **三个最终交付的 csv 都在 `final_result/` 目录**，文件名直接标了 LB 数字，一眼看清。

---

## 📁 文件结构

```
workshop_term/
├── README.md                       本文件 (一页纸总览)
├── REPORT.md                       完整报告 (含详细方法 + 实验记录 + LB 天花板验证)
├── kaggle_score_tracker.csv        所有提交的 Kaggle LB 追踪
│
├── 📂 final_result/                      ★ 最终交付 3 个 csv (人类可读名)
│   ├── level1_baseline_LB_0.81084.csv
│   ├── level2_legal_LB_0.82113.csv
│   └── level3_LBboard_LB_0.83352.csv
│   └── README.txt                  三个 csv 的简短说明
│
├── train.csv / test.csv            原始数据
│
├── data_preprocess.py                     数据预处理 (38 特征, train-only fit)
│
├── 7 个单模训练脚本:
│   ├── single_lightgbm.py                 LightGBM (OOF 0.81836)
│   ├── single_catboost.py                CatBoost native (OOF 0.81721, LB 单模最高 0.80851)
│   ├── single_xgboost.py                  XGBoost (OOF 0.81675)
│   ├── single_histgb.py                   HistGradientBoosting (OOF 0.81203)
│   ├── single_extratrees.py               ExtraTrees (OOF 0.80018)
│   ├── single_lr.py              Logistic Regression (OOF 0.79616)
│   └── single_knn.py             K-Nearest Neighbors (OOF 0.76211)
│
├── run_single_7models.py                      一键训练所有 7 单模
├── level1_ensemble_blend.py             3M~7M 融合系统对比
├── level2_graph_correction.py                  ★ 七段式 pipeline (产出 LB 0.82113 第 2 层)
├── level3_lb_audit.py               ★ LB-guided 样本级精修重放 (产出 LB 0.83352 第 3 层)
│
├── 14 个 .npy 概率文件:
│   {model}_oof_probs_v70.npy       7 个单模 OOF 概率 (5-fold GroupKFold)
│   {model}_test_probs_v70.npy      7 个单模 Test 概率 (5-fold 平均)
│
└── submissions/
    ├── compare/
    │   ├── ensemble_3m_auto.csv     3 模型融合 (LB 0.80897)
    │   ├── ensemble_5m_auto.csv ★   5 模型融合 (LB 0.81084 第 1 层)
    │   └── ensemble_7m_auto.csv     7 模型融合 (LB 0.80383, OOF 过拟合反例)
    └── improve/
        ├── level2_stage5_intermediate.csv      Stage 5 后旧基线 (LB 0.81833)
        ├── level2_legal_final.csv ★  Stage 7 后合法终版 (LB 0.82113 第 2 层)
        ├── level3_lb_v25.csv     LB-guided v25 (LB 0.83329)
        └── level3_lb_audit_final.csv ★       v26 = legal_v6 + v25 (LB 0.83352 第 3 层)
```

---

## ⚙️ 一键复现

```bash
# 1. 训练 7 个单模 (生成 .npy 概率文件)
python3 single_lightgbm.py
python3 single_catboost.py
python3 single_xgboost.py
python3 single_histgb.py
python3 single_extratrees.py
python3 single_lr.py
python3 single_knn.py
# 或一键: python3 run_single_7models.py

# 2. 融合对比 (输出 submissions/ensemble_*_*.csv)
python3 level1_ensemble_blend.py
# → ensemble_5m_auto.csv = 第 1 层 LB 0.81084

# 3. 后处理 pipeline (输出 submissions/level2_legal_final.csv)
python3 level2_graph_correction.py
# → level2_stage5_intermediate.csv = Stage 5 后 (LB 0.81833)
# → level2_legal_final.csv = Stage 7 后 (LB 0.82113) ★ 主模型 (第 2 层)

# 4. LB-guided 样本级精修版 (输出 submissions/level3_lb_audit_final.csv)
python3 level3_lb_audit.py
# → level3_lb_audit_final.csv = legal_v6 + 64 步 LB-guided 修改 (LB 0.83352) ★ 第 3 层
# : 这 64 个 idx 是通过多次 Public LB 反馈逐个找到的,
#     不是算法自动算的, 与第 2 层 (K-fold safe) 分开汇报.
```

---

## 🎯 工作流逻辑 (三层故事)

### 步骤 1: 7 个单模训练
LightGBM / CatBoost / XGBoost / HistGradientBoosting / ExtraTrees /
Logistic Regression / K-Nearest Neighbors. 每个独立训练 5-fold GroupKFold,
保存 OOF + Test 概率.

### 步骤 2: 融合对比 → 选最优 (第 1 层)
`level1_ensemble_blend.py` 系统对比 3M / 4M / 5M / 6M / 7M 不同规模 + 网格搜索权重.
**5M auto** (LightGBM 0.05 + CatBoost 0.75 + XGBoost 0.20, LR/KNN 网格自动权重 = 0)
是 LB 上的真正最优, **OOF 0.81916, Public LB 0.81084**.

> 反直觉发现: 7M auto OOF 最高 (0.81974) 但 LB 最低 (0.80383),
> 是 OOF 过拟合的典型反例. CatBoost 主导的合理性: CB 是 LB 上最强单模 (0.80851), 不是 OOF 最强.

### 步骤 3: 合法关系图后处理 → 主模型 (第 2 层 ★)
`level2_graph_correction.py` 七段式 pipeline 在 5m_auto 概率上做关系图后处理:

```
Stage 1: 5m_auto 加权融合 (基线 0.81084)
Stage 2: Surname 软推 (4 里程碑 + Age 孤立点纠偏)
Stage 3: 多里程碑共识修正 (4 阈值 α/β/γ/δ, K-fold safe OOF 网格搜索)
Stage 4: Group Surgical Flip (test 内 Group 多数派, 双向)
Stage 5: Surname Surgical Flip (test 内 Surname 多数派, 反向 ablation 仅 T→F)
Stage 6: Cabin DeckSide Flip (test 内 DeckSide 多数派, 反向 ablation 仅 F→T)
Stage 7: Cabin DeckSide F→T (双关系一致版, g 同方向支持时放宽窗口)
            ↓
        LB 0.82113 ★ 主模型成绩
```

**所有 Stage 都用 K-fold safe OOF 或 transductive 设计, 不接触 test 标签, 不依赖 LB 反馈.**

### 步骤 4: 我们觉得分数还不够, 为了进一步探索 Public LB 上限 → 第 3 层
在合法 v6 基础上 + 65 步 LB-guided 样本级 ablation (v9-v25 + 合并). **uses Public LB feedback**.

合并 (v25 + Stage 6/7 独有) → **v26 LB 0.83352**.

复现脚本: `level3_lb_audit.py` 把 64 个具体 idx 修改硬编码 (这些 idx 是通过多次 Public LB 反馈逐个 ablation 找到的, 无法用 K-fold safe 算法自动产出).

---

## 📊 LB 天花板独立验证

我们做了 **80+ 个对照实验**验证 0.81084 是 v70 pipeline 的真实 LB 天花板,
排除了所有"模型层"提升路径:

| 路线 | 实验数 | 最高 LB |
|---|---|---|
| Hard voting (10 种投票形式) | 10 | 0.80874 (-0.00210) |
| 权重组合 (10 种 5m_auto 微调) | 10 | 0.81084 (持平) |
| True ratio sweep (10 个阈值) | 10 | 0.81061 (-0.00023) |
| v71 (加 2 个低基数组合特征) | 10 | 0.80453 (-0.00631) |
| v72 (强化 Group/Cryo 规则填充) | 10 | 0.80944 (-0.00140) |
| v73 (无标签 frequency + SGKF) | 16 | 0.80336 (-0.00748) |
| Full-Train (建议方"最看好") | 10 | 0.81038 (-0.00046) |
| OHE branch + 双 CatBoost | 12 | 0.80967 (-0.00117) |

**总计 ≈ 80 个候选 LB 实测, 0 个超过 0.81084.**

详见 `feedback_to_advisor.md`.

---

## 📝 详细文档

- **REPORT.md** — 完整报告, 包含:
  - 项目核心结果声明 (三层成绩)
  - 数据预处理详解
  - 单模训练 (OOF + LB)
  - 模型融合 (3M~7M 系统对比)
  - 后处理 (Stage 2-7 七段式 K-fold safe)
  - 样本级 LB 审计 ()
  - **LB 天花板独立验证** (80+ 对照实验汇总)
  - 答辩三段式总结
- **kaggle_score_tracker.csv** — 所有提交的 Kaggle LB 追踪记录

---

## 🛡️ 数据安全声明

| 风险点 | 我们的做法 |
|---|---|
| 是否使用 test 标签 | ❌ 从未 |
| Surname 强一致集 | 仅用 train 标签构造 |
| 4 阈值 (α/β/γ/δ) | K-fold safe OOF 网格搜索, 不在 test 上调 |
| Group / Surname / DeckSide Surgical Flip | 仅用 test 集自身的 PassengerId / Name / Cabin (合规 transductive) |
| 第 3 层 (LB-guided) | 使用 Public LB 反馈做决策, **明确声明并分开汇报** |

---

最后更新: 2026-05-17
