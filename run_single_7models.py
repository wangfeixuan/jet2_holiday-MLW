"""
v7.0 一键Training 5 base models (LGB + CB + XGB + LR + KNN)
=================================================
所有模型 5 fold GroupKFold by PassengerId Group, 同一组超参 (来自 v6.7 实验).

Output .npy 概率文件 (供 level1_ensemble_blend.py blend使用):
  • lgbm_oof_probs_v70.npy  / lgbm_test_probs_v70.npy
  • catboost_oof_probs_v70.npy / catboost_test_probs_v70.npy
  • xgb_oof_probs_v70.npy   / xgb_test_probs_v70.npy
  • lr_oof_probs_v70.npy    / lr_test_probs_v70.npy
  • knn_oof_probs_v70.npy   / knn_test_probs_v70.npy

不再Generatingblend提交; blend统一交给 level1_ensemble_blend.py.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb_mod
from catboost import CatBoostClassifier, Pool
import xgboost as xgb_mod

from data_preprocess import get_unified_processed_data, get_linear_model_data

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

N_SPLITS = 5
THRESHOLD = 0.5

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'binary_error', 'boosting_type': 'gbdt',
    'verbose': -1, 'n_jobs': -1, 'bagging_freq': 5,
    'learning_rate': 0.0882, 'num_leaves': 25, 'max_depth': 10,
    'min_child_samples': 21, 'reg_alpha': 0.00226, 'reg_lambda': 0.10407,
    'subsample': 0.85866, 'colsample_bytree': 0.81394, 'seed': 42,
}
CB_PARAMS = {
    'iterations': 3000, 'eval_metric': 'Accuracy', 'early_stopping_rounds': 150,
    'verbose': 0, 'bootstrap_type': 'Bernoulli',
    'learning_rate': 0.08819, 'depth': 9, 'l2_leaf_reg': 2.19660,
    'min_data_in_leaf': 100, 'random_strength': 0.27593, 'subsample': 0.94276,
    'random_seed': 42,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'error',
    'tree_method': 'hist', 'enable_categorical': True, 'max_cat_to_onehot': 10,
    'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 3,
    'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'subsample': 0.85, 'colsample_bytree': 0.85,
    'n_jobs': -1, 'verbosity': 0, 'seed': 42,
}


def _align_categoricals(X, X_test, cat_cols):
    """让 train/test 共享同一套 category, 避免 test 出现新categories报错."""
    X = X.copy(); X_test = X_test.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).astype('category')
        X_test[c] = X_test[c].astype(str).astype('category')
        all_cat = pd.api.types.union_categoricals([X[c], X_test[c]]).categories
        X[c] = pd.Categorical(X[c], categories=all_cat)
        X_test[c] = pd.Categorical(X_test[c], categories=all_cat)
    return X, X_test


def train_lgb(X, y, X_test, groups, cat_cols):
    X, X_test = _align_categoricals(X, X_test, cat_cols)
    oof = np.zeros(len(X)); test = np.zeros(len(X_test))
    for tr, va in GroupKFold(N_SPLITS).split(X, y, groups=groups):
        d_tr = lgb_mod.Dataset(X.iloc[tr], y.iloc[tr], categorical_feature=cat_cols)
        d_va = lgb_mod.Dataset(X.iloc[va], y.iloc[va], categorical_feature=cat_cols, reference=d_tr)
        m = lgb_mod.train(LGB_PARAMS, d_tr, num_boost_round=3000, valid_sets=[d_va],
                          callbacks=[lgb_mod.early_stopping(150, verbose=False),
                                     lgb_mod.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        test += m.predict(X_test, num_iteration=m.best_iteration) / N_SPLITS
    return oof, test


def train_cb(X, y, X_test, groups, cat_cols):
    X = X.copy(); X_test = X_test.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str); X_test[c] = X_test[c].astype(str)
    oof = np.zeros(len(X)); test = np.zeros(len(X_test))
    for tr, va in GroupKFold(N_SPLITS).split(X, y, groups=groups):
        m = CatBoostClassifier(**CB_PARAMS)
        m.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=cat_cols),
              eval_set=Pool(X.iloc[va], y.iloc[va], cat_features=cat_cols),
              use_best_model=True)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test += m.predict_proba(X_test)[:, 1] / N_SPLITS
    return oof, test


def train_xgb(X, y, X_test, groups, cat_cols):
    X, X_test = _align_categoricals(X, X_test, cat_cols)
    oof = np.zeros(len(X)); test = np.zeros(len(X_test))
    d_test = xgb_mod.DMatrix(X_test, enable_categorical=True)
    for tr, va in GroupKFold(N_SPLITS).split(X, y, groups=groups):
        d_tr = xgb_mod.DMatrix(X.iloc[tr], y.iloc[tr], enable_categorical=True)
        d_va = xgb_mod.DMatrix(X.iloc[va], y.iloc[va], enable_categorical=True)
        m = xgb_mod.train(XGB_PARAMS, d_tr, num_boost_round=3000,
                          evals=[(d_va, 'va')], early_stopping_rounds=150, verbose_eval=False)
        rng = (0, m.best_iteration + 1)
        oof[va] = m.predict(d_va, iteration_range=rng)
        test += m.predict(d_test, iteration_range=rng) / N_SPLITS
    return oof, test


def train_lr(X_lin, y, X_test_lin, groups):
    oof = np.zeros(len(X_lin)); test = np.zeros(len(X_test_lin))
    for tr, va in GroupKFold(N_SPLITS).split(X_lin, y, groups=groups):
        m = LogisticRegression(C=0.1, penalty='l1', solver='liblinear',
                                max_iter=2000, random_state=42)
        m.fit(X_lin[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X_lin[va])[:, 1]
        test += m.predict_proba(X_test_lin)[:, 1] / N_SPLITS
    return oof, test


def train_knn(X_lin, y, X_test_lin, groups):
    oof = np.zeros(len(X_lin)); test = np.zeros(len(X_test_lin))
    for tr, va in GroupKFold(N_SPLITS).split(X_lin, y, groups=groups):
        m = KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1)
        m.fit(X_lin[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X_lin[va])[:, 1]
        test += m.predict_proba(X_test_lin)[:, 1] / N_SPLITS
    return oof, test


def train_et(X_lin, y, X_test_lin, groups):
    """ExtraTrees with 3-seed averaging for stability."""
    from sklearn.ensemble import ExtraTreesClassifier
    seeds = [42, 2024, 7]
    y_arr = y.values if hasattr(y, 'values') else y
    oof_avg = np.zeros(len(X_lin)); test_avg = np.zeros(len(X_test_lin))
    for seed in seeds:
        oof = np.zeros(len(X_lin)); test = np.zeros(len(X_test_lin))
        for tr, va in GroupKFold(N_SPLITS).split(X_lin, y_arr, groups=groups):
            m = ExtraTreesClassifier(n_estimators=500, max_features='sqrt',
                                     min_samples_leaf=10, min_samples_split=20,
                                     random_state=seed, n_jobs=-1)
            m.fit(X_lin[tr], y_arr[tr])
            oof[va] = m.predict_proba(X_lin[va])[:, 1]
            test += m.predict_proba(X_test_lin)[:, 1] / N_SPLITS
        oof_avg += oof / len(seeds)
        test_avg += test / len(seeds)
    return oof_avg, test_avg


def train_hgb(X, y, X_test, groups, cat_cols):
    """HistGradientBoosting with ordinal-encoded cats, 3-seed averaging."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    X_enc = X.copy(); X_test_enc = X_test.copy()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value',
                         unknown_value=-1, encoded_missing_value=-1)
    X_cat = X[cat_cols].astype(str).fillna('__MISSING__')
    X_test_cat = X_test[cat_cols].astype(str).fillna('__MISSING__')
    enc.fit(pd.concat([X_cat, X_test_cat], axis=0))
    X_enc[cat_cols] = enc.transform(X_cat)
    X_test_enc[cat_cols] = enc.transform(X_test_cat)
    X_enc = X_enc.astype(float).values
    X_test_enc = X_test_enc.astype(float).values
    y_arr = y.values if hasattr(y, 'values') else y
    seeds = [42, 2024, 3407]
    oof_avg = np.zeros(len(X_enc)); test_avg = np.zeros(len(X_test_enc))
    for seed in seeds:
        oof = np.zeros(len(X_enc)); test = np.zeros(len(X_test_enc))
        for tr, va in GroupKFold(N_SPLITS).split(X_enc, y_arr, groups=groups):
            m = HistGradientBoostingClassifier(
                max_iter=1000, learning_rate=0.05, max_depth=8,
                min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=50, random_state=seed)
            m.fit(X_enc[tr], y_arr[tr])
            oof[va] = m.predict_proba(X_enc[va])[:, 1]
            test += m.predict_proba(X_test_enc)[:, 1] / N_SPLITS
        oof_avg += oof / len(seeds)
        test_avg += test / len(seeds)
    return oof_avg, test_avg


print("=" * 60)
print("v7.0 Training 7 base models (LGB + CB + XGB + LR + KNN + ET + HGB)")
print("=" * 60)

t_all = time.time()
X, y, X_test, cat_cols = get_unified_processed_data()
X_lin, _, X_test_lin, _ = get_linear_model_data()
groups = pd.read_csv('train.csv')['PassengerId'].apply(lambda x: x.split('_')[0]).values

print(f"\nData: tree models X={X.shape}, linear X_lin={X_lin.shape}, categories={len(cat_cols)}")

JOBS = [
    ("LightGBM",   lambda: train_lgb(X, y, X_test, groups, cat_cols), 'lgbm'),
    ("CatBoost",   lambda: train_cb(X, y, X_test, groups, cat_cols),  'catboost'),
    ("XGBoost",    lambda: train_xgb(X, y, X_test, groups, cat_cols), 'xgb'),
    ("LogReg",     lambda: train_lr(X_lin, y, X_test_lin, groups),    'lr'),
    ("KNN",        lambda: train_knn(X_lin, y, X_test_lin, groups),   'knn'),
    ("ExtraTrees", lambda: train_et(X_lin, y, X_test_lin, groups),    'extratrees'),
    ("HistGB",     lambda: train_hgb(X, y, X_test, groups, cat_cols), 'histgb'),
]

for name, fn, prefix in JOBS:
    t = time.time()
    print(f"   {name} ...", end='', flush=True)
    oof, test = fn()
    np.save(f'{prefix}_oof_probs_v70.npy', oof)
    np.save(f'{prefix}_test_probs_v70.npy', test)
    acc = accuracy_score(y, (oof > THRESHOLD).astype(int))
    print(f" OOF={acc:.5f}  ({time.time()-t:.1f}s)  -> {prefix}_*_v70.npy")

print(f"\nTotal elapsed: {(time.time()-t_all)/60:.1f} min")
print("7 base model probabilities saved. Now run level1_ensemble_blend.py for ensemble comparison.")
print("=" * 60)
