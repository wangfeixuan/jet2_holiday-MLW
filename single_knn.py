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

from sklearn.neighbors import KNeighborsClassifier
from data_preprocess import get_linear_model_data

def run_knn(X=None, y=None, X_test=None, groups=None, save_outputs=True):
    if X is None:
        X, y, X_test, _ = get_linear_model_data()
        groups = pd.read_csv('train.csv')['PassengerId'].apply(lambda x: x.split('_')[0]).values
    oof = np.zeros(len(X)); test = np.zeros(len(X_test)); last = None
    for tr, va in GroupKFold(N_SPLITS).split(X, y, groups=groups):
        m = KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1)
        m.fit(X[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]; test += m.predict_proba(X_test)[:, 1] / N_SPLITS
        last = m
    t0 = time.perf_counter(); _ = last.predict_proba(X_test)[:, 1]; infer = time.perf_counter() - t0
    if save_outputs: _save('knn', 'single_knn.csv', oof, test)
    return oof, test, infer

if __name__ == '__main__':
    run_knn()
