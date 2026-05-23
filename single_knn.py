"""
Spaceship Titanic - K 近邻分类器 (sklearn KNeighborsClassifier)
===============================================================
目的：为集成提供"几何上完全不同"的diversity来源。
- KNN 用距离决策，跟树（分stage）和 LR（linear）几何上完全不同
- 经验上 KNN ↔ tree models correlation 0.85~0.90，比 LR (0.93) diversity更强
- 单模分数预期 ~0.78（比树低），但集成增益可能很可观

调参思路：
- 自动in [10, 20, 30, 50, 75, 100] 中扫 k，选 OOF 最佳的（通过 5 折 CV validation）
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report

# === 数据版本开关 ===
DATA_VERSION = "v7.0"
from data_preprocess import get_unified_processed_data, get_linear_model_data

import os
import time
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
os.makedirs("submissions", exist_ok=True)


def run_knn():
    print("=" * 60)
    print(f"Starting KNN Training (DATA={DATA_VERSION})")
    print("=" * 60)
    start_time = time.time()

    # ========== 1. 获取预Processing数据 ==========
    # KNN 必须用 StandardScaler 后的数据，否则量纲大的特征会主导距离
    X, y, X_test, feature_names = get_linear_model_data()
    print(f"\nData info:")
    print(f"   Train shape:    {X.shape}")
    print(f"   Test shape:    {X_test.shape}")
    # print(f"   Total features:      {X.shape[1]} (one-hot + 标准化后)")

    # ========== 2. GroupKFold ==========
    train_df = pd.read_csv(os.path.join(_HERE, "train.csv"))
    groups = train_df['PassengerId'].apply(lambda x: x.split('_')[0]).values

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    # ========== 3. 自动调 k 和 weights ==========
    # 扩大 k searched范围（之前 k=150 时曲线还in涨，可能更大 k 更好）
    # 同时comparison weights='uniform' vs 'distance'
    k_candidates = [50, 100, 200, 300, 500, 800]
    weights_candidates = ['uniform', 'distance']
    # print(f"\n扫描 k ∈ {k_candidates} × weights ∈ {weights_candidates}")
    # print(f"   共 {len(k_candidates) * len(weights_candidates)} combinations")

    # 提前把所有折的索引算出来，避免重复计算
    fold_splits = list(gkf.split(X, y, groups=groups))

    best_config = None
    best_oof_score = 0
    best_oof_probs = None
    best_test_probs = None
    all_results = []

    for weights in weights_candidates:
        for k in k_candidates:
            oof_probs = np.zeros(len(X))
            test_probs = np.zeros(len(X_test))

            for fold, (train_idx, val_idx) in enumerate(fold_splits):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]

                model = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weights,
                    n_jobs=-1,
                )
                model.fit(X_tr, y_tr)
                oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
                test_probs += model.predict_proba(X_test)[:, 1] / n_splits

            score = accuracy_score(y, (oof_probs > 0.5).astype(int))
            all_results.append((k, weights, score))
            marker = "" if score > best_oof_score else "  "
            print(f"   {marker} k={k:>3d}  weights={weights:<8s}  OOF accuracy: {score:.4f}")

            if score > best_oof_score:
                best_oof_score = score
                best_config = (k, weights)
                best_oof_probs = oof_probs.copy()
                best_test_probs = test_probs.copy()

    best_k, best_weights = best_config
    print(f"\nOptimal config: k={best_k}, weights={best_weights}, OOF accuracy: {best_oof_score:.4f}")
    k_results = [(k, w, s) for k, w, s in all_results]

    # ========== 4. Saved OOF / Test 概率 ==========
    ver_suffix = DATA_VERSION.replace(".", "")
    np.save(f'knn_oof_probs_{ver_suffix}.npy', best_oof_probs)
    np.save(f'knn_test_probs_{ver_suffix}.npy', best_test_probs)
    print(f"\nOOF/Test probabilities saved (for ensemble)")

    # ========== 5. 总体 OOF 评估 ==========
    THRESHOLD = 0.5
    final_oof_score = accuracy_score(y, (best_oof_probs > THRESHOLD).astype(int))
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"TrainingDone!")
    print(f"{'=' * 60}")
    print(f"   All combination results:")
    for k, w, s in sorted(k_results, key=lambda x: -x[2]):
        print(f"     k={k:>3d}, weights={w:<8s}: {s:.4f}")
    print(f"   Optimal config:      k={best_k}, weights={best_weights}")
    print(f"   Threshold used:     {THRESHOLD}")
    print(f"   Final OOF accuracy: {final_oof_score:.4f}")
    print(f"   Total elapsed:       {elapsed:.1f} s")
    print(f"{'=' * 60}")

    # ========== 6. GeneratingSubmission file ==========
    test_df = pd.read_csv(os.path.join(_HERE, "test.csv"))
    final_test_preds = (best_test_probs > THRESHOLD).astype(bool)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_test_preds,
    })
    output_filename = "submissions/single_knn.csv"
    submission.to_csv(output_filename, index=False)
    print(f"\nSubmission file: {os.path.abspath(output_filename)}")
    print(f"   Predicted Transported: {final_test_preds.sum()}/{len(final_test_preds)} "
          f"({final_test_preds.mean() * 100:.1f}%)")

    # ========== 7. OOF classification report ==========
    print(f"\nOOF classification report:")
    print(classification_report(
        y, (best_oof_probs > THRESHOLD).astype(int),
        target_names=['Not Transported', 'Transported']
    ))

    # ========== 8. vsother modelscorrelation分析 ==========
    print(f"\nKNN vs other models OOF correlation:")
    correlations = {}
    for name, path in [
        ('LightGBM', f'lgbm_oof_probs_{ver_suffix}.npy'),
        ('CatBoost', f'catboost_oof_probs_{ver_suffix}.npy'),
        ('XGBoost',  f'xgb_oof_probs_{ver_suffix}.npy'),
        ('LR',       f'lr_oof_probs_{ver_suffix}.npy'),
        ('MLP',      f'mlp_oof_probs_{ver_suffix}.npy'),
    ]:
        if os.path.exists(path):
            other_oof = np.load(path)
            corr = np.corrcoef(best_oof_probs, other_oof)[0, 1]
            correlations[name] = corr
            print(f"   KNN ↔ {name:10s}: {corr:.4f}")

    if correlations:
        avg_corr = np.mean(list(correlations.values()))
        print(f"\n   Mean correlation: {avg_corr:.4f}")
        if avg_corr < 0.88:
            print(f"   Correlation very low！KNN provides excellent diversity, ensemble will improve significantly!")
        elif avg_corr < 0.92:
            print(f"   Correlation low！KNN diversity should bring noticeable improvement")
        elif avg_corr < 0.95:
            print(f"   Correlation medium，ensemble should improve noticeably")
        else:
            print(f"   Correlation high，insufficient diversity")

    print(f"\n{'=' * 60}")
    print(f"All done! Submission file for Kaggle: {output_filename}")
    print(f"{'=' * 60}")

    return best_oof_probs, best_test_probs, final_oof_score


if __name__ == "__main__":
    run_knn()
