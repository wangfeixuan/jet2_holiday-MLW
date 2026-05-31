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
    Each rerun may have ~0.0002 level differences due to multi-threaded floating-point drift, so single-model outputs are isolated to npy_new/."""
    os.makedirs('npy_new', exist_ok=True)
    os.makedirs('submissions/npy_new', exist_ok=True)
    np.save(f'npy_new/{prefix}_oof_probs_{VERSION}.npy', oof)
    np.save(f'npy_new/{prefix}_test_probs_{VERSION}.npy', test)
    ids = pd.read_csv('test.csv')['PassengerId']
    pd.DataFrame({'PassengerId': ids, 'Transported': (test > THRESHOLD).astype(bool)}).to_csv(
        os.path.join('submissions/npy_new', submission_name), index=False)

from catboost import CatBoostClassifier, Pool
from data_preprocess import get_unified_processed_data

CB_PARAMS = {
    'iterations': 3000, 'eval_metric': 'Accuracy', 'early_stopping_rounds': 150,
    'verbose': 0, 'bootstrap_type': 'Bernoulli', 'learning_rate': 0.08819,
    'depth': 9, 'l2_leaf_reg': 2.19660, 'min_data_in_leaf': 100,
    'random_strength': 0.27593, 'subsample': 0.94276, 'random_seed': 42,
}

def run_catboost(X=None, y=None, X_test=None, groups=None, cat_cols=None, save_outputs=True):
    if X is None:
        X, y, X_test, cat_cols = get_unified_processed_data()
        groups = pd.read_csv('train.csv')['PassengerId'].apply(lambda x: x.split('_')[0]).values
    X = X.copy(); X_test = X_test.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str); X_test[c] = X_test[c].astype(str)
    oof = np.zeros(len(X)); test = np.zeros(len(X_test)); last = None
    for tr, va in GroupKFold(N_SPLITS).split(X, y, groups=groups):
        m = CatBoostClassifier(**CB_PARAMS)
        m.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=cat_cols), eval_set=Pool(X.iloc[va], y.iloc[va], cat_features=cat_cols), use_best_model=True)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test += m.predict_proba(X_test)[:, 1] / N_SPLITS
        last = m
    t0 = time.perf_counter(); _ = last.predict_proba(X_test)[:, 1]; infer = time.perf_counter() - t0
    if save_outputs: _save('catboost', 'single_catboost.csv', oof, test)
    return oof, test, infer

if __name__ == '__main__':
    run_catboost()
