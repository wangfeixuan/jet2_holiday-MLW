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
    """Save to npy_new/ to avoid overwriting the authoritative npy files in the root directory.
    The *_v70.npy files in the root directory are the verified optimal probabilities. ensemble/Level2/Level3 always read from this set, similar to the fixed 5m weights.
    each rerun由于 XGBoost hist multi-threaded floating-point drift会有 ~0.0002 level differences, 所以single-model outputsisolated to npy_new/。"""
    os.makedirs('npy_new', exist_ok=True)
    os.makedirs('submissions/npy_new', exist_ok=True)
    np.save(f'npy_new/{prefix}_oof_probs_{VERSION}.npy', oof)
    np.save(f'npy_new/{prefix}_test_probs_{VERSION}.npy', test)
    ids = pd.read_csv('test.csv')['PassengerId']
    pd.DataFrame({'PassengerId': ids, 'Transported': (test > THRESHOLD).astype(bool)}).to_csv(
        os.path.join('submissions/npy_new', submission_name), index=False)

import lightgbm as lgb_mod
from data_preprocess import get_unified_processed_data

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'binary_error', 'boosting_type': 'gbdt',
    'verbose': -1, 'n_jobs': -1, 'bagging_freq': 5,
    'learning_rate': 0.0882, 'num_leaves': 25, 'max_depth': 10,
    'min_child_samples': 21, 'reg_alpha': 0.00226, 'reg_lambda': 0.10407,
    'subsample': 0.85866, 'colsample_bytree': 0.81394, 'seed': 42,
}

def _align(X, X_test, cat_cols):
    X = X.copy(); X_test = X_test.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).astype('category')
        X_test[c] = X_test[c].astype(str).astype('category')
        cats = pd.api.types.union_categoricals([X[c], X_test[c]]).categories
        X[c] = pd.Categorical(X[c], categories=cats)
        X_test[c] = pd.Categorical(X_test[c], categories=cats)
    return X, X_test

def run_lightgbm(X=None, y=None, X_test=None, groups=None, cat_cols=None, save_outputs=True):
    if X is None:
        X, y, X_test, cat_cols = get_unified_processed_data()
        groups = pd.read_csv('train.csv')['PassengerId'].apply(lambda x: x.split('_')[0]).values
    X, X_test = _align(X, X_test, cat_cols)
    oof = np.zeros(len(X)); test = np.zeros(len(X_test)); last = None; last_iter = 0
    for tr, va in GroupKFold(N_SPLITS).split(X, y, groups=groups):
        dtr = lgb_mod.Dataset(X.iloc[tr], y.iloc[tr], categorical_feature=cat_cols)
        dva = lgb_mod.Dataset(X.iloc[va], y.iloc[va], categorical_feature=cat_cols, reference=dtr)
        m = lgb_mod.train(LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb_mod.early_stopping(150, verbose=False), lgb_mod.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        test += m.predict(X_test, num_iteration=m.best_iteration) / N_SPLITS
        last = m; last_iter = m.best_iteration
    t0 = time.perf_counter(); _ = last.predict(X_test, num_iteration=last_iter); infer = time.perf_counter() - t0
    if save_outputs: _save('lgbm', 'single_lightgbm.csv', oof, test)
    return oof, test, infer

if __name__ == '__main__':
    run_lightgbm()
