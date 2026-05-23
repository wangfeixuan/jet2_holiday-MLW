"""
Spaceship Titanic - LightGBM advanced v1.1 (Bug Fix)
============================================
修复内容：
1. thresholdsearched改用 np.linspace（避免浮点数精度 bug）
2. 移除 sns.barplot 的 xerr 参数（新版 seaborn 不支持）
3. 移除 sample_submission.csv 依赖，自动从 test.csv Generating提交
4. TrainingDone后立即Saved概率和Submission file，画图Failed不影响主结果
5. 所有可视化用 try-except 包裹，确保Submission file一定能Generating
 
特性：
1. 多 seed 平均（3 不同种子）降低方差
2. 5 折分layer交叉validation
3. Categorical features用 LightGBM 原生支持（category dtype）
4. 自动Saved OOF/Test 概率，供后续 Stacking 使用
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
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
 
 
def run_enhanced_lightgbm():
    print("=" * 60)
    print("Starting LightGBM advanced v1.1 Training")
    print("=" * 60)
    start_time = time.time()
 
    # ========== 1. 获取预Processing数据 ==========
    X, y, X_test_raw, cat_cols = get_unified_processed_data()
 
    # LightGBM 需要 category dtype
    for col in cat_cols:
        X[col] = X[col].astype(str).astype('category')
        X_test_raw[col] = X_test_raw[col].astype(str).astype('category')
 
    # train 和 test categories对齐（避免 test 出现新categories报错）
    for col in cat_cols:
        all_categories = pd.api.types.union_categoricals(
            [X[col], X_test_raw[col]]
        ).categories
        X[col] = pd.Categorical(X[col], categories=all_categories)
        X_test_raw[col] = pd.Categorical(X_test_raw[col], categories=all_categories)
 
    print(f"\nData info:")
    print(f"   Train shape: {X.shape}")
    print(f"   Test shape: {X_test_raw.shape}")
    print(f"   Total features:   {X.shape[1]}")
    print(f"   Categorical features:   {len(cat_cols)}")
 
    # ========== 2. LightGBM 参数 ==========
    lgb_params = {
    # --- 基础vs环境参数（保持不变） ---
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'n_jobs': -1,
    'bagging_freq': 5, 
    
    # --- Optuna 跑出的最佳参数 ---
    'learning_rate': 0.08820,
    'num_leaves': 25,
    'max_depth': 10,
    'min_child_samples': 21,
    'reg_alpha': 0.00226,       # 等同于旧代码的 lambda_l1
    'reg_lambda': 0.10407,      # 等同于旧代码的 lambda_l2
    'subsample': 0.85866,       # 等同于旧代码的 bagging_fraction
    'colsample_bytree': 0.81394 # 等同于旧代码的 feature_fraction
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
 # ========== 4. 多 Seed × K 折 Training循环 (GroupKFold 防作弊版) ==========
    
    # 【新增逻辑】：单独读取 train.csv，提取出用于划分的 groups 数组
    train_df = pd.read_csv(os.path.join(_HERE, "train.csv"))
    groups = train_df['PassengerId'].apply(lambda x: x.split('_')[0])
    
    for seed_idx, seed in enumerate(seeds):
        print(f"{'=' * 60}")
        print(f"Seed {seed_idx + 1}/{len(seeds)}: random_seed = {seed}")
        print(f"{'=' * 60}")

        lgb_params['seed'] = seed
        lgb_params['feature_fraction_seed'] = seed
        lgb_params['bagging_seed'] = seed
        
        # 【修改逻辑】：使用 GroupKFold，它不需要 random_state，因为它是按群组绝对划分的
        gkf = GroupKFold(n_splits=n_splits)

        oof_probs_seed = np.zeros(len(X))
        test_probs_seed = np.zeros(len(X_test_raw))

        # 【修改逻辑】：in split 中传入 groups 参数，强制同组同折
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
 
            train_data = lgb.Dataset(
                X_tr, label=y_tr,
                categorical_feature=cat_cols
            )
            val_data = lgb.Dataset(
                X_val, label=y_val,
                categorical_feature=cat_cols,
                reference=train_data
            )
 
            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=5000,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
 
            fold_counter += 1
            feature_importances[f'model_{fold_counter}'] = model.feature_importance(
                importance_type='gain'
            )
 
            val_p = model.predict(X_val, num_iteration=model.best_iteration)
            oof_probs_seed[val_idx] = val_p
            test_probs_seed += model.predict(
                X_test_raw, num_iteration=model.best_iteration
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
    np.save(f'lgbm_oof_probs_{DATA_VERSION.replace(".", "")}.npy', oof_probs_final)
    np.save(f'lgbm_test_probs_{DATA_VERSION.replace(".", "")}.npy', test_probs_final)
    print(f"\nOOF/Test probabilities saved (for stacking)")
 
    # ========== 6. 寻找optimalthreshold（修复版：用 linspace） ==========
    print(f"\n{'=' * 60}")
    # print("寻找optimal分类threshold...")
    print(f"{'=' * 60}")
 
    best_threshold = 0.5
    best_score = 0
    threshold_history = []
 
    thresholds_to_try = np.linspace(0.30, 0.70, 81)
    for threshold in thresholds_to_try:
        preds = (oof_probs_final > threshold).astype(int)
        score = accuracy_score(y, preds)
        threshold_history.append((threshold, score))
        if score > best_score:
            best_score = score
            best_threshold = threshold
 
    sorted_history = sorted(threshold_history, key=lambda x: -x[1])[:5]
    print(f"   Top 5 threshold candidates:")
    for thr, sc in sorted_history:
        print(f"     threshold {thr:.4f} → OOF accuracy {sc:.4f}")
 
    # ========== 7. Output最终结果 ==========
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"TrainingDone!")
    print(f"{'=' * 60}")
    print(f"   Per-seed OOF scores: {[f'{s:.4f}' for s in seed_scores]}")
    print(f"   Seed mean OOF:    {np.mean(seed_scores):.4f}")
    print(f"   Seed std:      {np.std(seed_scores):.4f}")
    print(f"   Best threshold:      {best_threshold:.4f}")
    print(f"   Final OOF accuracy: {best_score:.4f}")
    print(f"   Total elapsed:        {elapsed / 60:.1f}  min")
    print(f"{'=' * 60}")
 
    # ========== 8. 立刻GeneratingSubmission file（最重要！） ==========
    print(f"\nGeneratingSubmission file...")
    test_df = pd.read_csv(os.path.join(_HERE, "test.csv"))
 
    # 同时Generatingoptimalthreshold版本和默认 0.5 threshold版本（方便comparison）
    final_test_preds_best = (test_probs_final > best_threshold).astype(bool)
    final_test_preds_05 = (test_probs_final > 0.5).astype(bool)
 
    # optimalthreshold版本
    submission_best = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_test_preds_best
    })
    output_filename_best = "submissions/single_lightgbm_best_threshold.csv"
    submission_best.to_csv(output_filename_best, index=False)
 
    # 默认threshold 0.5 版本
    submission_05 = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_test_preds_05
    })
    output_filename_05 = "submissions/single_lightgbm.csv"
    submission_05.to_csv(output_filename_05, index=False)
 
    print(f"   optimalthresholdSubmission file: {os.path.abspath(output_filename_best)}")
    print(f"      Predicted True: {final_test_preds_best.sum()} / {len(final_test_preds_best)} "
          f"({final_test_preds_best.mean() * 100:.1f}%)")
    # print(f"   默认 0.5 threshold文件: {os.path.abspath(output_filename_05)}")
    print(f"      Predicted True: {final_test_preds_05.sum()} / {len(final_test_preds_05)} "
          f"({final_test_preds_05.mean() * 100:.1f}%)")
    print(f"   Submit both and compare Kaggle scores")
 
    # ========== 9. 可视化（用 try-except 包裹） ==========
    print(f"\nGenerating visualizations...")
 
    # 9.1 特征重要性图（修复：移除 xerr）
    try:
        feature_importances['average'] = feature_importances.mean(axis=1)
        feature_importances = feature_importances.sort_values(by='average', ascending=False)
 
        plt.figure(figsize=(12, 10))
        top_n = min(30, len(feature_importances))
        top_features = feature_importances.head(top_n)
        sns.barplot(
            x=top_features['average'],
            y=top_features.index,
            palette='magma'
        )
        plt.title(f'LightGBM Top {top_n} Feature Importance (avg over {n_total_models} models)',
                  fontsize=14)
        plt.xlabel('Importance Score (Gain)')
        plt.tight_layout()
        plt.savefig('lgbm_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Feature importance plot saved: lgbm_feature_importance.png")
    except Exception as e:
        print(f"   Feature importance plot savedFailed: {e}")
 
    # 9.2 threshold优化曲线
    try:
        thresholds, scores = zip(*threshold_history)
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, scores, linewidth=2, color='purple')
        plt.axvline(best_threshold, color='red', linestyle='--',
                    label=f'Best: {best_threshold:.3f} ({best_score:.4f})')
        plt.title('LightGBM Threshold Optimization Curve')
        plt.xlabel('Threshold')
        plt.ylabel('OOF Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('lgbm_threshold_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Threshold curve saved: lgbm_threshold_curve.png")
    except Exception as e:
        print(f"   Threshold curve savedFailed: {e}")
 
    # 9.3 混淆矩阵
    try:
        final_oof_preds = (oof_probs_final > best_threshold).astype(int)
        cm = confusion_matrix(y, final_oof_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=['Not Transported', 'Transported'],
                    yticklabels=['Not Transported', 'Transported'])
        plt.title(f'LightGBM OOF Confusion Matrix (Accuracy: {best_score:.4f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('lgbm_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Confusion matrix saved: lgbm_confusion_matrix.png")
 
        print(f"\nOOF classification report:")
        print(classification_report(y, final_oof_preds,
                                     target_names=['Not Transported', 'Transported']))
    except Exception as e:
        print(f"   Confusion matrix savedFailed: {e}")
 
    print(f"\n{'=' * 60}")
    print(f"All done!")
    # print(f"   推荐Submission file for Kaggle: {output_filename_best}")
    print(f"   Alternative file (default threshold):     {output_filename_05}")
    print(f"{'=' * 60}")
 
    return oof_probs_final, test_probs_final, best_threshold, best_score
 
 
if __name__ == "__main__":
    run_enhanced_lightgbm()