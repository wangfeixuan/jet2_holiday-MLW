"""
Spaceship Titanic - HistGradientBoosting v7.0
============================================

目的：
1. 不改 data_preprocess.py
2. Training HistGradientBoostingClassifier
3. Output：
   - submissions/histgb_0.5_v70.csv
   - histgb_oof_probs_v70.npy
   - histgb_test_probs_v70.npy
"""

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder

from data_preprocess import get_unified_processed_data


# ========== 基础设置 ==========
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
os.makedirs("submissions", exist_ok=True)

VERSION = "v70"
N_SPLITS = 5
SEEDS = [42, 2024, 3407]
THRESHOLD = 0.5

print("=" * 70)
print("HistGradientBoosting v7.0 training")
print("=" * 70)


# ========== 1. 读取数据 ==========
X, y, X_test, cat_cols = get_unified_processed_data()
test_df = pd.read_csv("test.csv")

# print(f"Training集: {X.shape}")
print(f"Test set: {X_test.shape}")
print(f"categorical features: {len(cat_cols)}")
print(f"numeric features: {X.shape[1] - len(cat_cols)}")


# ========== 2. Categorical features编码 ==========
X_enc = X.copy()
X_test_enc = X_test.copy()

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-1
)

X_cat = X[cat_cols].astype(str).fillna("__MISSING__")
X_test_cat = X_test[cat_cols].astype(str).fillna("__MISSING__")

# fit train + test 的categories空间，只为避免 test 新categories报错，不使用标签
encoder.fit(pd.concat([X_cat, X_test_cat], axis=0))

X_enc[cat_cols] = encoder.transform(X_cat)
X_test_enc[cat_cols] = encoder.transform(X_test_cat)

X_enc = X_enc.astype(float)
X_test_enc = X_test_enc.astype(float)

y_array = y.values if hasattr(y, "values") else y


# ========== 3. 多 seed + 5 折Training ==========
oof_probs = np.zeros(len(X_enc))
test_probs_all = []
fold_scores = []

for seed in SEEDS:
    print("\n" + "-" * 70)
    print(f"Seed = {seed}")
    print("-" * 70)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    seed_oof = np.zeros(len(X_enc))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_enc, y_array), 1):
        X_tr = X_enc.iloc[tr_idx]
        X_val = X_enc.iloc[val_idx]
        y_tr = y_array[tr_idx]
        y_val = y_array[val_idx]

        model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.035,
            max_leaf_nodes=31,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=0.05,
            random_state=seed * 100 + fold
        )

        model.fit(X_tr, y_tr)

        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test_enc)[:, 1]

        seed_oof[val_idx] = val_prob
        test_probs_all.append(test_prob)

        fold_acc = accuracy_score(y_val, val_prob > THRESHOLD)
        fold_scores.append(fold_acc)

        print(f"Fold {fold}: acc = {fold_acc:.5f}")

    seed_acc = accuracy_score(y_array, seed_oof > THRESHOLD)
    print(f"Seed {seed} OOF acc = {seed_acc:.5f}")

    oof_probs += seed_oof / len(SEEDS)


# ========== 4. 汇总 ==========
test_probs = np.mean(test_probs_all, axis=0)

final_oof_acc = accuracy_score(y_array, oof_probs > THRESHOLD)
test_preds = test_probs > THRESHOLD

print("\n" + "=" * 70)
print("HistGradientBoosting v7.0 result")
print("=" * 70)
print(f"Fold mean acc: {np.mean(fold_scores):.5f}")
print(f"Final OOF acc: {final_oof_acc:.5f}")
print(f"Test True ratio: {test_preds.mean() * 100:.2f}%")

# ========== 5. Saved概率文件 ==========
np.save(f"histgb_oof_probs_{VERSION}.npy", oof_probs)
np.save(f"histgb_test_probs_{VERSION}.npy", test_probs)

print(f"Saved: histgb_oof_probs_{VERSION}.npy")
print(f"Saved: histgb_test_probs_{VERSION}.npy")


# ========== 6. Saved Kaggle Submission file ==========
out_path = "submissions/single_histgb.csv"

pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Transported": test_preds.astype(bool)
}).to_csv(out_path, index=False)

print(f"Saved submission: {out_path}")
print("=" * 70)