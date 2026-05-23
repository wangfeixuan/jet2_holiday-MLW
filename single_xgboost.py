"""
Spaceship Titanic - XGBoost advanced v2.0 (rewritten)
============================================
v2.0 修复关键问题：
修复1: API 混用 - 之前混用了 sklearn API 参数 (n_estimators) 
          和 native API Training (xgb.train)，导致部分参数被静默忽略
修复2: 弃用 OneHot, 改用 XGBoost 1.6+ 的原生categories支持
          (enable_categorical=True), 减少特征维度, vs LGB/CB 行为更一致
修复3: 重置参数 - 旧参数是基于 v5.0 数据 + Standard KFold 调出来的，
          gamma=3.98 in新数据上过度正则化, 改用合理的默认参数
特性：
1. 多 seed 平均（3 不同种子）降低方差
2. GroupKFold 5 折交叉validation（防止家族泄漏）
3. 原生Categorical features支持
4. 自动Saved OOF/Test 概率，供后续 Stacking 使用
5. 固定 0.5 threshold
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
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
os.chdir(_HERE)  # 让后续所有相对路径Output都落in workshop_term/
os.makedirs("submissions", exist_ok=True)  # Submission file统一放in submissions/ 子文件夹
pd.set_option('future.no_silent_downcasting', True)
plt.style.use('ggplot')
 
# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
 
 
def run_enhanced_xgboost():
    print("=" * 60)
    print("Starting XGBoost advanced v2.0 Training（rewritten）")
    print("=" * 60)
    start_time = time.time()
 
    # ========== 1. 获取预Processing数据 ==========
    X, y, X_test_raw, cat_cols = get_unified_processed_data()
 
    # ========== 防泄漏核心逻辑 ==========
    train_df = pd.read_csv(os.path.join(_HERE, "train.csv"))
    groups = train_df['PassengerId'].apply(lambda x: x.split('_')[0])
 
    # 删除 ID 类特征
    drop_cols = [col for col in ['Group', 'PP'] if col in X.columns]
    X = X.drop(columns=drop_cols)
    X_test_raw = X_test_raw.drop(columns=drop_cols)
    cat_cols = [col for col in cat_cols if col in X.columns]
    # ======================================================
 
    # ========== 【关键修改】使用 XGBoost 原生categories支持 ==========
    # 不再 OneHot，改用 category dtype，让 XGB 原生Processing
    # print(f"\n转换Categorical features为 category dtype（XGBoost 原生支持）...")
    for col in cat_cols:
        # 注意：要先把 train 和 test 的categories合并，确保 dtype 一致
        all_categories = pd.concat([X[col].astype(str), X_test_raw[col].astype(str)]).unique()
        X[col] = pd.Categorical(X[col].astype(str), categories=all_categories)
        X_test_raw[col] = pd.Categorical(X_test_raw[col].astype(str), categories=all_categories)
 
    print(f"\nData info:")
    print(f"   Train shape: {X.shape}")
    print(f"   Test shape: {X_test_raw.shape}")
    print(f"   Total features:   {X.shape[1]}")
    print(f"   Categorical features:   {len(cat_cols)}")
 
    # ========== 2. XGBoost 参数（重新设置合理默认值） ==========
    # 不再使用旧调参结果，因为那是为 v5.0 + Standard KFold 调的
    # 用一组保守但合理的默认参数，等以后再做 Optuna 调参
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'tree_method': 'hist',
        'enable_categorical': True,    # 启用原生categories支持
        'max_cat_to_onehot': 10,       # categories数 < 10 用 OneHot, 否则用 Target Encoding
        'learning_rate': 0.05,
        'max_depth': 6,                # 中等深度
        'min_child_weight': 3,
        'gamma': 0.1,                  # 极小的剪枝threshold（旧参数 gamma=3.98 太激进）
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'n_jobs': -1,
        'verbosity': 0,
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
 
    # print(f"\nStarting training: {len(seeds)}  seed × {n_splits} 折 = {n_total_models} 模型\n")
 
    # ========== 4. 多 Seed × K 折 Training循环 ==========
    for seed_idx, seed in enumerate(seeds):
        print(f"{'=' * 60}")
        print(f"Seed {seed_idx + 1}/{len(seeds)}: random_seed = {seed}")
        print(f"{'=' * 60}")
 
        xgb_params['seed'] = seed
        gkf = GroupKFold(n_splits=n_splits)
 
        oof_probs_seed = np.zeros(len(X))
        test_probs_seed = np.zeros(len(X_test_raw))
 
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
 
            # ========== 【关键修改】使用 enable_categorical=True ==========
            dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
            dtest = xgb.DMatrix(X_test_raw, enable_categorical=True)
 
            # ========== 【关键修改】统一使用 native API ==========
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=5000,
                evals=[(dtrain, 'train'), (dval, 'valid')],
                early_stopping_rounds=200,
                verbose_eval=False
            )
 
            fold_counter += 1
            # 记录特征重要性（用 gain）
            importance_dict = model.get_score(importance_type='gain')
            for feat in X.columns:
                feature_importances.loc[feat, f'model_{fold_counter}'] = importance_dict.get(feat, 0)
 
            # Predicted
            val_p = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            oof_probs_seed[val_idx] = val_p
            test_probs_seed += model.predict(
                dtest, iteration_range=(0, model.best_iteration + 1)
            ) / n_splits
 
            fold_acc = accuracy_score(y_val, (val_p > 0.5).astype(int))
            print(f"   Fold {fold + 1}/{n_splits} validation accuracy: {fold_acc:.4f} "
                  f"(最佳迭代: {model.best_iteration})")
 
        seed_acc = accuracy_score(y, (oof_probs_seed > 0.5).astype(int))
        seed_scores.append(seed_acc)
        print(f"\n   Seed {seed} OOF accuracy: {seed_acc:.4f}\n")
 
        oof_probs_final += oof_probs_seed / len(seeds)
        test_probs_final += test_probs_seed / len(seeds)
 
    # ========== 5. 立刻Saved OOF / Test 概率 ==========
    np.save(f'xgb_oof_probs_{DATA_VERSION.replace(".", "")}.npy', oof_probs_final)
    np.save(f'xgb_test_probs_{DATA_VERSION.replace(".", "")}.npy', test_probs_final)
    print(f"\nOOF/Test probabilities saved (for stacking)")
 
    # ========== 6. 直接用 0.5 threshold ==========
    THRESHOLD = 0.5
    final_oof_score = accuracy_score(y, (oof_probs_final > THRESHOLD).astype(int))
 
    # ========== 7. Output最终结果 ==========
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"TrainingDone!")
    print(f"{'=' * 60}")
    print(f"   Per-seed OOF scores: {[f'{s:.4f}' for s in seed_scores]}")
    print(f"   Seed mean OOF:    {np.mean(seed_scores):.4f}")
    print(f"   Seed std:      {np.std(seed_scores):.4f}")
    # print(f"   Threshold used:      {THRESHOLD}（默认值）")
    print(f"   Final OOF accuracy: {final_oof_score:.4f}")
    print(f"   Total elapsed:        {elapsed / 60:.1f}  min")
    print(f"{'=' * 60}")
 
    # ========== 8. 立刻GeneratingSubmission file ==========
    print(f"\nGeneratingSubmission file...")
    test_df = pd.read_csv(os.path.join(_HERE, "test.csv"))
    final_test_preds = (test_probs_final > THRESHOLD).astype(bool)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_test_preds
    })
 
    output_filename = "submissions/single_xgboost.csv"
    submission.to_csv(output_filename, index=False)
    abs_path = os.path.abspath(output_filename)
    print(f"   Submission fileSaved: {abs_path}")
    print(f"   Predicted Transported: {final_test_preds.sum()} / {len(final_test_preds)} "
          f"({final_test_preds.mean() * 100:.1f}%)")
 
    # ========== 9. 可视化 ==========
    print(f"\nGenerating visualizations...")
 
    try:
        feature_importances['average'] = feature_importances.mean(axis=1)
        feature_importances = feature_importances.sort_values(by='average', ascending=False)
 
        plt.figure(figsize=(12, 10))
        top_n = min(30, len(feature_importances))
        top_features = feature_importances.head(top_n)
        sns.barplot(
            x=top_features['average'],
            y=top_features.index,
            palette='rocket'
        )
        plt.title(f'XGBoost v2.0 Top {top_n} Feature Importance',
                  fontsize=14)
        plt.xlabel('Importance Score (Gain)')
        plt.tight_layout()
        plt.savefig('xgb_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Feature importance plot saved: xgb_feature_importance.png")
    except Exception as e:
        print(f"   Feature importance plot savedFailed: {e}")
 
    try:
        final_oof_preds = (oof_probs_final > THRESHOLD).astype(int)
        cm = confusion_matrix(y, final_oof_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Not Transported', 'Transported'],
                    yticklabels=['Not Transported', 'Transported'])
        plt.title(f'XGBoost v2.0 OOF Confusion Matrix (Accuracy: {final_oof_score:.4f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('xgb_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Confusion matrix saved: xgb_confusion_matrix.png")
 
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
    run_enhanced_xgboost()