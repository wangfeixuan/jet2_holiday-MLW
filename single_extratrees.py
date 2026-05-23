"""
Spaceship Titanic - ExtraTrees Classifier
============================================
ExtraTrees (极度随机树) - 比 RandomForest 更随机的集成
- 每分裂点的threshold随机选取（而非 RF 的"optimal分裂"）
- 因此具有更强的"反过拟合"特性
- 跟 LightGBM/CatBoost/XGBoost 算法layer面差异大，能注入新信号

参数选择:
- n_estimators=500 (够多但不过分)
- max_features='sqrt' (经典选择)
- min_samples_leaf=10 (温和正则)
- 5 折 GroupKFold + 3  seed = 15 模型
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report

DATA_VERSION = "v7.0"

if DATA_VERSION == "v6.4":
    from data_v64 import get_linear_model_data
elif DATA_VERSION == "v6.5":
    from data_v65 import get_linear_model_data
elif DATA_VERSION == "v6.6":
    from data_v66 import get_linear_model_data
elif DATA_VERSION == "v6.7":
    from data_v67 import get_linear_model_data
elif DATA_VERSION == "v7.0":
    from data_preprocess import get_linear_model_data
else:
    raise ValueError(f"只支持 v6.4 / v6.5 / v6.6 / v6.7 / v7.0，当前 DATA_VERSION={DATA_VERSION}")

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
os.makedirs("submissions", exist_ok=True)

# ========== 1. 数据 ==========
print("=" * 60)
print(f"Starting ExtraTrees Training (DATA={DATA_VERSION})")
print("=" * 60)

start_time = time.time()
X, y, X_test, _ = get_linear_model_data()
y = y.values if hasattr(y, 'values') else y
print(f"\nData info:")
print(f"   Train shape: {X.shape}")
print(f"   Test shape: {X_test.shape}")

# ========== 2. GroupKFold ==========
train_df = pd.read_csv(os.path.join(_HERE, "train.csv"))
groups = train_df['PassengerId'].apply(lambda x: x.split('_')[0]).values

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

# ========== 3. Training ==========
seeds = [42, 2024, 7]
# print(f"\nStarting training: {len(seeds)}  seed × {n_splits} 折 = {len(seeds)*n_splits} 模型\n")

oof_probs_avg = np.zeros(len(X))
test_probs_avg = np.zeros(len(X_test))
seed_oof_scores = []

for seed_idx, seed in enumerate(seeds):
    print("=" * 60)
    print(f"Seed {seed_idx+1}/{len(seeds)}: random_state = {seed}")
    print("=" * 60)

    oof_probs = np.zeros(len(X))
    test_probs = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = ExtraTreesClassifier(
            n_estimators=500,
            max_features='sqrt',
            min_samples_leaf=10,
            min_samples_split=20,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        oof_probs[va_idx] = model.predict_proba(X_va)[:, 1]
        test_probs += model.predict_proba(X_test)[:, 1] / n_splits

        fold_acc = accuracy_score(y_va, (oof_probs[va_idx] > 0.5).astype(int))
        print(f"   Fold {fold+1}/{n_splits} validation accuracy: {fold_acc:.4f}")

    seed_oof_acc = accuracy_score(y, (oof_probs > 0.5).astype(int))
    seed_oof_scores.append(seed_oof_acc)
    print(f"\n   Seed {seed} OOF accuracy: {seed_oof_acc:.4f}\n")

    oof_probs_avg += oof_probs / len(seeds)
    test_probs_avg += test_probs / len(seeds)

# ========== 4. Saved OOF/Test 概率 ==========
ver_suffix = DATA_VERSION.replace(".", "")
# ensemble2.py 期望的文件名前缀是 extratrees_
np.save(f'extratrees_oof_probs_{ver_suffix}.npy', oof_probs_avg)
np.save(f'extratrees_test_probs_{ver_suffix}.npy', test_probs_avg)
# 同时保留旧命名兼容性
np.save(f'et_oof_probs_{ver_suffix}.npy', oof_probs_avg)
np.save(f'et_test_probs_{ver_suffix}.npy', test_probs_avg)
print("OOF/Test probabilities saved (for ensemble)")

# ========== 5. 总体 OOF 评估 ==========
final_oof_acc = accuracy_score(y, (oof_probs_avg > 0.5).astype(int))
elapsed = time.time() - start_time

print(f"\n{'=' * 60}")
print(f"TrainingDone!")
print(f"{'=' * 60}")
print(f"   Per-seed OOF scores: {[f'{s:.4f}' for s in seed_oof_scores]}")
print(f"   Seed mean OOF:    {np.mean(seed_oof_scores):.4f}")
print(f"   Seed std:      {np.std(seed_oof_scores):.4f}")
print(f"   Final OOF accuracy: {final_oof_acc:.4f}")
print(f"   Total elapsed:        {elapsed:.1f} s")
print(f"{'=' * 60}")

# ========== 6. GeneratingSubmission file ==========
test_df = pd.read_csv(os.path.join(_HERE, "test.csv"))
preds = (test_probs_avg > 0.5).astype(bool)
fname = "submissions/single_extratrees.csv"
pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': preds,
}).to_csv(fname, index=False)
print(f"\nSubmission file: {os.path.abspath(fname)}")
print(f"   Predicted Transported: {preds.sum()}/{len(preds)} ({preds.mean()*100:.1f}%)")

# ========== 7. vsother modelscorrelation ==========
print(f"\nExtraTrees vs other models OOF correlation:")
prefix_map = {'LightGBM':'lgbm', 'CatBoost':'catboost', 'XGBoost':'xgb',
              'MLP':'mlp', 'LR':'lr', 'KNN':'knn'}
for name, prefix in prefix_map.items():
    fname_oof = f"{prefix}_oof_probs_{ver_suffix}.npy"
    if os.path.exists(fname_oof):
        other_oof = np.load(fname_oof)
        corr = np.corrcoef(oof_probs_avg, other_oof)[0, 1]
        marker = "" if corr < 0.85 else ("  " if corr < 0.92 else "")
        print(f"   {marker} ExtraTrees ↔ {name:<10s}: {corr:.4f}")

print(f"\n{'=' * 60}")
print(f"All done!")
print(f"{'=' * 60}")
