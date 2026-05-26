import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

os.chdir(os.path.dirname(os.path.abspath(__file__)))
N_SPLITS = 5
THRESHOLD = 0.5
VERSION = 'v70'

def _save(prefix, submission_name, oof, test):
    """保存到 npy_new/ 不再覆盖根目录的权威 npy。
    根目录 *_v70.npy 是已验证的最优概率, ensemble/Level2/Level3 始终读这套, 类似 5m 写死权重。
    每次重跑由于 XGBoost hist 多线程浮点漂移会有 ~0.0002 级别差, 所以单模产物隔离到 npy_new/。"""
    os.makedirs('npy_new', exist_ok=True)
    os.makedirs('submissions/npy_new', exist_ok=True)
    np.save(f'npy_new/{prefix}_oof_probs_{VERSION}.npy', oof)
    np.save(f'npy_new/{prefix}_test_probs_{VERSION}.npy', test)
    ids = pd.read_csv('test.csv')['PassengerId']
    pd.DataFrame({'PassengerId': ids, 'Transported': (test > THRESHOLD).astype(bool)}).to_csv(
        os.path.join('submissions/npy_new', submission_name), index=False)

import xgboost as xgb_mod
from data_preprocess import get_unified_processed_data

XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'error', 'tree_method': 'hist',
    'enable_categorical': True, 'max_cat_to_onehot': 10, 'learning_rate': 0.05,
    'max_depth': 6, 'min_child_weight': 3, 'gamma': 0.1, 'reg_alpha': 0.1,
    'reg_lambda': 1.0, 'subsample': 0.85, 'colsample_bytree': 0.85,
    'n_jobs': -1, 'verbosity': 0, 'seed': 42,
}

def _align(X, X_test, cat_cols):
    """与旧版 run_v70.py::_align_categoricals 完全一致, 保证概率可复现。"""
    X = X.copy(); X_test = X_test.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).astype('category')
        X_test[c] = X_test[c].astype(str).astype('category')
        all_cat = pd.api.types.union_categoricals([X[c], X_test[c]]).categories
        X[c] = pd.Categorical(X[c], categories=all_cat)
        X_test[c] = pd.Categorical(X_test[c], categories=all_cat)
    return X, X_test

def run_xgboost(X=None, y=None, X_test=None, groups=None, cat_cols=None, save_outputs=True):
    """XGBoost 训练: 与旧版 run_v70.py::train_xgb 一致 (单 seed=42, 5 fold GroupKFold)。"""
    if X is None:
        X, y, X_test, cat_cols = get_unified_processed_data()
        groups = pd.read_csv('train.csv')['PassengerId'].apply(lambda x: x.split('_')[0]).values
    X, X_test = _align(X, X_test, cat_cols)
    oof = np.zeros(len(X)); test = np.zeros(len(X_test))
    dtest = xgb_mod.DMatrix(X_test, enable_categorical=True)
    last = None; last_rng = (0, 1)
    for tr, va in GroupKFold(N_SPLITS).split(X, y, groups=groups):
        dtr = xgb_mod.DMatrix(X.iloc[tr], y.iloc[tr], enable_categorical=True)
        dva = xgb_mod.DMatrix(X.iloc[va], y.iloc[va], enable_categorical=True)
        m = xgb_mod.train(XGB_PARAMS, dtr, num_boost_round=3000,
                          evals=[(dva, 'va')], early_stopping_rounds=150, verbose_eval=False)
        rng = (0, m.best_iteration + 1)
        oof[va] = m.predict(dva, iteration_range=rng)
        test += m.predict(dtest, iteration_range=rng) / N_SPLITS
        last = m; last_rng = rng
    t0 = time.perf_counter(); _ = last.predict(dtest, iteration_range=last_rng); infer = time.perf_counter() - t0
    if save_outputs: _save('xgb', 'single_xgboost.csv', oof, test)
    return oof, test, infer

if __name__ == '__main__':
    run_xgboost()
