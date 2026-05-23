"""
Spaceship Titanic - 逻辑回归 (sklearn LogisticRegression)
========================================================
目的：为集成提供"几何上互补"的diversity来源。
- LR 学linear决策边界，vs GBDT 的分stage边界完全不同
- 单模分数预期 ~0.79（比树低），但vs LGB/CB/XGB Correlation low
- Trainings级Done，性价比极高
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report

# === 数据版本开关：改这一行同时切换数据来源 + Output文件名 ===
DATA_VERSION = "v7.0"
from data_preprocess import get_unified_processed_data, get_linear_model_data

import os
import time
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
os.makedirs("submissions", exist_ok=True)


def run_logistic_regression():
    print("=" * 60)
    print(f"Starting Logistic Regression Training (DATA={DATA_VERSION})")
    print("=" * 60)
    start_time = time.time()

    # ========== 1. 获取预Processing数据 ==========
    # get_linear_model_data() 返回的是经 one-hot + StandardScaler 后的 numpy 矩阵
    X, y, X_test, feature_names = get_linear_model_data()
    print(f"\nData info:")
    print(f"   Train shape:    {X.shape}")
    print(f"   Test shape:    {X_test.shape}")
    # print(f"   Total features:      {X.shape[1]} (one-hot 后)")

    # ========== 2. GroupKFold (跟other models对齐) ==========
    # 同 PassengerId 前缀（同 Group）的人必须in同一折，避免组内信息泄漏到validation集
    train_df = pd.read_csv(os.path.join(_HERE, "train.csv"))
    groups = train_df['PassengerId'].apply(lambda x: x.split('_')[0]).values

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    # ========== 3. LR 参数 ==========
    # 改为 L1 + 小 C：让 LR 自动选择重要特征、决策更稀疏，
    #     使 LR vstree models的correlation下降（目标 < 0.92），增强集成diversity。
    # solver='liblinear' 支持 L1（lbfgs 不支持 L1）
    lr_params = dict(
        C=0.1,
        penalty='l1',
        solver='liblinear',
        max_iter=2000,
        random_state=42,
    )
    # print(f"\nLR params: C={lr_params['C']}, penalty={lr_params['penalty']}, "
    #       f"solver={lr_params['solver']}")

    # ========== 4. K 折Training ==========
    # 注：LR + lbfgs 是完全确定性的，不需要 multi-seed（不像tree models有随机采样）
    oof_probs = np.zeros(len(X))
    test_probs = np.zeros(len(X_test))
    fold_scores = []

    # print(f"\nStarting training: {n_splits} 折 (LR 无随机性，单次足够)\n")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

        model = LogisticRegression(**lr_params)
        model.fit(X_tr, y_tr)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test)[:, 1] / n_splits

        fold_acc = accuracy_score(y_val, (val_probs > 0.5).astype(int))
        fold_scores.append(fold_acc)
        print(f"   Fold {fold + 1}/{n_splits} validation accuracy: {fold_acc:.4f}")

    # ========== 5. Saved OOF / Test 概率 ==========
    ver_suffix = DATA_VERSION.replace(".", "")
    np.save(f'lr_oof_probs_{ver_suffix}.npy', oof_probs)
    np.save(f'lr_test_probs_{ver_suffix}.npy', test_probs)
    print(f"\nOOF/Test probabilities saved (for ensemble)")

    # ========== 6. 总体 OOF 评估 ==========
    THRESHOLD = 0.5
    final_oof_score = accuracy_score(y, (oof_probs > THRESHOLD).astype(int))
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"TrainingDone!")
    print(f"{'=' * 60}")
    print(f"   Per-fold accuracy: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"   Fold mean:      {np.mean(fold_scores):.4f}")
    print(f"   Fold std:    {np.std(fold_scores):.4f}")
    print(f"   Threshold used:     {THRESHOLD}")
    print(f"   Final OOF accuracy: {final_oof_score:.4f}")
    print(f"   Total elapsed:       {elapsed:.1f} s")
    print(f"{'=' * 60}")

    # ========== 7. GeneratingSubmission file ==========
    test_df = pd.read_csv(os.path.join(_HERE, "test.csv"))
    final_test_preds = (test_probs > THRESHOLD).astype(bool)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_test_preds,
    })
    output_filename = "submissions/single_lr.csv"
    submission.to_csv(output_filename, index=False)
    print(f"\nSubmission file: {os.path.abspath(output_filename)}")
    print(f"   Predicted Transported: {final_test_preds.sum()}/{len(final_test_preds)} "
          f"({final_test_preds.mean() * 100:.1f}%)")

    # ========== 8. OOF classification report ==========
    print(f"\nOOF classification report:")
    print(classification_report(
        y, (oof_probs > THRESHOLD).astype(int),
        target_names=['Not Transported', 'Transported']
    ))

    # ========== 9. vs other models OOF correlation分析（关键！） ==========
    # correlation越低，集成增益越大。目标 < 0.93
    print(f"\nLR vs other models OOF correlation:")
    correlations = {}
    for name, path in [
        ('LightGBM', f'lgbm_oof_probs_{ver_suffix}.npy'),
        ('CatBoost', f'catboost_oof_probs_{ver_suffix}.npy'),
        ('XGBoost',  f'xgb_oof_probs_{ver_suffix}.npy'),
        ('MLP',      f'mlp_oof_probs_{ver_suffix}.npy'),
    ]:
        if os.path.exists(path):
            other_oof = np.load(path)
            corr = np.corrcoef(oof_probs, other_oof)[0, 1]
            correlations[name] = corr
            print(f"   LR ↔ {name:10s}: {corr:.4f}")

    if correlations:
        avg_corr = np.mean(list(correlations.values()))
        print(f"\n   Mean correlation: {avg_corr:.4f}")
        if avg_corr < 0.90:
            print(f"   Correlation low! LR provides excellent diversity.")
        elif avg_corr < 0.93:
            print(f"   Correlation medium-low, LR should bring noticeable improvement.")
        elif avg_corr < 0.96:
            print(f"   Correlation medium, ensemble may improve slightly.")
        else:
            print(f"   Correlation high, insufficient diversity, ensemble improvement limited.")

    print(f"\n{'=' * 60}")
    print(f"All done! Submission file for Kaggle: {output_filename}")
    print(f"{'=' * 60}")
    print(f"\nNext step：")
    print(f"   Run ensemble for LGB+CB+XGB+LR 4-model weighted blend")
    print(f"{'=' * 60}")

    return oof_probs, test_probs, final_oof_score


if __name__ == "__main__":
    run_logistic_regression()
