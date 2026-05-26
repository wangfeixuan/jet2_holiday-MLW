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
    每次重跑由于多线程浮点漂移会有 ~0.0002 级别差, 所以单模产物隔离到 npy_new/。"""
    os.makedirs('npy_new', exist_ok=True)
    os.makedirs('submissions/npy_new', exist_ok=True)
    np.save(f'npy_new/{prefix}_oof_probs_{VERSION}.npy', oof)
    np.save(f'npy_new/{prefix}_test_probs_{VERSION}.npy', test)
    ids = pd.read_csv('test.csv')['PassengerId']
    pd.DataFrame({'PassengerId': ids, 'Transported': (test > THRESHOLD).astype(bool)}).to_csv(
        os.path.join('submissions/npy_new', submission_name), index=False)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from data_preprocess import get_unified_processed_data

def run_histgb(X=None, y=None, X_test=None, groups=None, cat_cols=None, save_outputs=True):
    if X is None:
        X, y, X_test, cat_cols = get_unified_processed_data()
        groups = pd.read_csv('train.csv')['PassengerId'].apply(lambda x: x.split('_')[0]).values
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)
    Xe = X.copy(); Xte = X_test.copy()
    Xcat = X[cat_cols].astype(str).fillna('__MISSING__'); Xtcat = X_test[cat_cols].astype(str).fillna('__MISSING__')
    enc.fit(pd.concat([Xcat, Xtcat], axis=0))
    Xe[cat_cols] = enc.transform(Xcat); Xte[cat_cols] = enc.transform(Xtcat)
    Xe = Xe.astype(float).values; Xte = Xte.astype(float).values
    y_arr = y.values if hasattr(y, 'values') else y
    seeds = [42, 2024, 3407]
    oof_avg = np.zeros(len(Xe)); test_avg = np.zeros(len(Xte)); last = None
    for seed in seeds:
        oof = np.zeros(len(Xe)); test = np.zeros(len(Xte))
        for tr, va in GroupKFold(N_SPLITS).split(Xe, y_arr, groups=groups):
            m = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.05, max_depth=8, min_samples_leaf=20, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=50, random_state=seed)
            m.fit(Xe[tr], y_arr[tr])
            oof[va] = m.predict_proba(Xe[va])[:, 1]; test += m.predict_proba(Xte)[:, 1] / N_SPLITS
            last = m
        oof_avg += oof / len(seeds); test_avg += test / len(seeds)
    t0 = time.perf_counter(); _ = last.predict_proba(Xte)[:, 1]; infer = time.perf_counter() - t0
    if save_outputs: _save('histgb', 'single_histgb.csv', oof_avg, test_avg)
    return oof_avg, test_avg, infer

if __name__ == '__main__':
    run_histgb()
