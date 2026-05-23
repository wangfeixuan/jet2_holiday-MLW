"""
Spaceship Titanic - CatBoost advanced v2.2 (最终稳定版)
============================================
关键变更：
1. 移除threshold优化（实测发现threshold优化in OOF 上有效但Test set上反而掉分）
2. 直接用 0.5 默认threshold（vs原始 baseline 一致，避免threshold过拟合）
3. 同时Savedoptimalthreshold版本（仅供参考，不推荐提交）
 
Bug 修复：
1. 移除 sample_submission.csv 依赖，自动从 test.csv Generating提交
2. 移除 sns.barplot 的 xerr 参数（新版 seaborn 不支持）
3. TrainingDone后立即Saved概率和Submission file，画图Failed不影响主结果
4. 所有可视化用 try-except 包裹
 
特性：
1. 多 seed 平均（3 不同种子）降低方差
2. 学习率 0.03 + 迭代 5000 + 早停 200
3. 自动Saved OOF/Test 概率，供后续 Stacking 使用
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# === 数据版本开关：改这一行同时切换数据来源 + Output文件名 ===
DATA_VERSION = "v7.0"
from data_preprocess import get_unified_processed_data, get_linear_model_data
import warnings
import time
import os
 
# 环境设置
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))  # 脚本自己的目录，用于定位 csv
os.chdir(_HERE)  # 让后续所有相对路径Output（概率 npy、提交 csv、图片）都落in workshop_term/
os.makedirs("submissions", exist_ok=True)  # Submission file统一放in submissions/ 子文件夹
pd.set_option('future.no_silent_downcasting', True)
plt.style.use('ggplot')
 
# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
 
 
def run_enhanced_catboost():
    print("=" * 60)
    print("Starting CatBoost advanced v2.2 Training")
    print("=" * 60)
    start_time = time.time()
 
    # ========== 1. 获取预Processing数据 ==========
    X, y, X_test_raw, cat_cols = get_unified_processed_data()

    # ========== 防泄漏核心逻辑 ==========
    # 从原始 train.csv 提取 Group 分组
    train_df = pd.read_csv(os.path.join(_HERE, "train.csv"))
    groups = train_df['PassengerId'].apply(lambda x: x.split('_')[0])

    # 删除会导致数据泄漏的 ID 特征
    drop_cols = [col for col in ['Group', 'PP'] if col in X.columns]
    X = X.drop(columns=drop_cols)
    X_test_raw = X_test_raw.drop(columns=drop_cols)

    # 过滤Categorical features列表，确保vs当前 X 匹配
    cat_cols = [col for col in cat_cols if col in X.columns]
    # ======================================================

    for col in cat_cols:
        X[col] = X[col].astype(str)
        X_test_raw[col] = X_test_raw[col].astype(str)
 
    print(f"\n Data info:")
    print(f"   Train shape: {X.shape}")
    print(f"   Test shape: {X_test_raw.shape}")
    print(f"   Total features:   {X.shape[1]}")
    print(f"   Categorical features:   {len(cat_cols)}")
 
    # ========== 2. CatBoost 参数 ==========
    cb_params = {
        'iterations': 5000,
        'eval_metric': 'Accuracy',
        'early_stopping_rounds': 200,
        'verbose': 0,
        'bootstrap_type': 'Bernoulli',
        'learning_rate': 0.08819,
        'depth': 9,
        'l2_leaf_reg': 2.19660,
        'min_data_in_leaf': 100,
        'random_strength': 0.27593,
        'subsample': 0.94276
    }
 
    # ========== 3. 多 seed 平均 ==========
    seeds = [42, 2024, 7]
    n_splits = 5
    n_total_models = len(seeds) * n_splits
 
    oof_probs_final = np.zeros(len(X))
    test_probs_final = np.zeros(len(X_test_raw))
 
    feature_importances = pd.DataFrame(index=X.columns)
    fold_counter = 0
    seed_scores = []
 
    # print(f"\n Starting training: {len(seeds)}  seed × {n_splits} 折 = {n_total_models} 模型\n")
 
    # ========== 4. 多 Seed × K 折 Training循环 ==========
    for seed_idx, seed in enumerate(seeds):
        print(f"{'=' * 60}")
        print(f" Seed {seed_idx + 1}/{len(seeds)}: random_seed = {seed}")
        print(f"{'=' * 60}")
 
        cb_params['random_seed'] = seed
        
        # 修改：GroupKFold 无 shuffle / random_state
        gkf = GroupKFold(n_splits=n_splits)
 
        oof_probs_seed = np.zeros(len(X))
        test_probs_seed = np.zeros(len(X_test_raw))
 
        # 修改：传入 groups=groups 防泄漏
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
 
            train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
            val_pool = Pool(X_val, y_val, cat_features=cat_cols)
 
            model = CatBoostClassifier(**cb_params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
 
            fold_counter += 1
            feature_importances[f'model_{fold_counter}'] = model.get_feature_importance()
 
            val_p = model.predict_proba(X_val)[:, 1]
            oof_probs_seed[val_idx] = val_p
            test_probs_seed += model.predict_proba(X_test_raw)[:, 1] / n_splits
 
            fold_acc = accuracy_score(y_val, (val_p > 0.5).astype(int))
            print(f"   Fold {fold + 1}/{n_splits} validation accuracy: {fold_acc:.4f} "
                  f"(最佳迭代: {model.get_best_iteration()})")
 
        seed_acc = accuracy_score(y, (oof_probs_seed > 0.5).astype(int))
        seed_scores.append(seed_acc)
        print(f"\n   Seed {seed} OOF accuracy: {seed_acc:.4f}\n")
 
        oof_probs_final += oof_probs_seed / len(seeds)
        test_probs_final += test_probs_seed / len(seeds)
 
    # ========== 5. 立刻Saved OOF / Test 概率 ==========
    np.save(f'catboost_oof_probs_{DATA_VERSION.replace(".", "")}.npy', oof_probs_final)
    np.save(f'catboost_test_probs_{DATA_VERSION.replace(".", "")}.npy', test_probs_final)
    print(f"\n OOF/Test probabilities saved (for stacking)")
 
    # ========== 6. 直接用 0.5 threshold（不再做thresholdsearched） ==========
    THRESHOLD = 0.5  # 固定使用 0.5，避免threshold过拟合
    final_oof_score = accuracy_score(y, (oof_probs_final > THRESHOLD).astype(int))
 
    # ========== 7. Output最终结果 ==========
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f" TrainingDone!")
    print(f"{'=' * 60}")
    print(f"   Per-seed OOF scores: {[f'{s:.4f}' for s in seed_scores]}")
    print(f"   Seed mean OOF:    {np.mean(seed_scores):.4f}")
    print(f"   Seed std:      {np.std(seed_scores):.4f}")
    # print(f"   Threshold used:      {THRESHOLD}（默认值）")
    print(f"   Final OOF accuracy: {final_oof_score:.4f}")
    print(f"   Total elapsed:        {elapsed / 60:.1f}  min")
    print(f"=" * 60)
 
    # ========== 8. 立刻GeneratingSubmission file ==========
    print(f"\n GeneratingSubmission file...")
    test_df = pd.read_csv(os.path.join(_HERE, "test.csv"))
    final_test_preds = (test_probs_final > THRESHOLD).astype(bool)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_test_preds
    })
 
    output_filename = "submissions/single_catboost.csv"
    submission.to_csv(output_filename, index=False)
    abs_path = os.path.abspath(output_filename)
    print(f"   Submission fileSaved: {abs_path}")
    print(f"   Predicted Transported: {final_test_preds.sum()} / {len(final_test_preds)} "
          f"({final_test_preds.mean() * 100:.1f}%)")
 
    # ========== 9. 可视化（用 try-except 包裹） ==========
    print(f"\nGenerating visualizations...")
 
    # 9.1 特征重要性图
    try:
        feature_importances['average'] = feature_importances.mean(axis=1)
        feature_importances = feature_importances.sort_values(by='average', ascending=False)
 
        plt.figure(figsize=(12, 10))
        top_n = min(30, len(feature_importances))
        top_features = feature_importances.head(top_n)
        sns.barplot(
            x=top_features['average'],
            y=top_features.index,
            palette='viridis'
        )
        plt.title(f'CatBoost Top {top_n} Feature Importance (avg over {n_total_models} models)',
                  fontsize=14)
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Feature importance plot saved: feature_importance.png")
    except Exception as e:
        print(f"   Feature importance plot savedFailed: {e}")
 
    # 9.2 混淆矩阵
    try:
        final_oof_preds = (oof_probs_final > THRESHOLD).astype(int)
        cm = confusion_matrix(y, final_oof_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Transported', 'Transported'],
                    yticklabels=['Not Transported', 'Transported'])
        plt.title(f'CatBoost OOF Confusion Matrix (Accuracy: {final_oof_score:.4f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Confusion matrix saved: confusion_matrix.png")
 
        print(f"\nOOF classification report:")
        print(classification_report(y, final_oof_preds,
                                     target_names=['Not Transported', 'Transported']))
    except Exception as e:
        print(f"   Confusion matrix savedFailed: {e}")
 
    print(f"\n{'=' * 60}")
    print(f"All done! Submission file for Kaggle: {output_filename}")
    print(f"{'=' * 60}")
 
    return oof_probs_final, test_probs_final, final_oof_score
 
 
if __name__ == "__main__":
    run_enhanced_catboost()