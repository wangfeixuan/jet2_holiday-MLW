"""Run all seven single-model scripts and collect metrics, memory, and logs."""
import os
import sys
import time
import warnings
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from data_preprocess import get_unified_processed_data, get_linear_model_data
from single_lightgbm import run_lightgbm
from single_catboost import run_catboost
from single_xgboost import run_xgboost
from single_lr import run_lr
from single_knn import run_knn
from single_extratrees import run_extratrees
from single_histgb import run_histgb

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('logs', exist_ok=True)
LOG_PATH = f"logs/run_single_7models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


_log_file = open(LOG_PATH, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)

try:
    import psutil
    PROCESS = psutil.Process(os.getpid())
except Exception:
    PROCESS = None


def rss_mb():
    if PROCESS is None:
        return np.nan
    return PROCESS.memory_info().rss / (1024 ** 2)


def measure_run(fn, interval=0.02):
    """跑模型并测时间/内存。返回 (elapsed, baseline, peak, delta, result)。
    result 是单模函数的返回值 (oof, test, infer_time)。"""
    baseline = rss_mb()
    peak = baseline
    running = True
    result = {'value': None}

    def sampler():
        nonlocal peak
        while running:
            current = rss_mb()
            if not np.isnan(current):
                peak = max(peak, current)
            time.sleep(interval)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result['value'] = fn()
    finally:
        running = False
        thread.join(timeout=1.0)
    elapsed = time.perf_counter() - start
    current = rss_mb()
    if not np.isnan(current):
        peak = max(peak, current)
    delta = peak - baseline if not np.isnan(peak) and not np.isnan(baseline) else np.nan
    return elapsed, baseline, peak, delta, result['value']


def compute_metrics(y_true, prob):
    pred = (prob > 0.5).astype(int)
    return {
        'Accuracy': accuracy_score(y_true, pred),
        'Precision': precision_score(y_true, pred),
        'Recall': recall_score(y_true, pred),
        'F1': f1_score(y_true, pred),
        'ROC_AUC': roc_auc_score(y_true, prob),
        'LogLoss': log_loss(y_true, prob),
    }


print('=' * 72)
print('v7.0 Running 7 single-model functions')
print('=' * 72)
print(f'Run log: {LOG_PATH}')

X, y, X_test, cat_cols = get_unified_processed_data()
X_lin, _, X_test_lin, _ = get_linear_model_data()
groups = pd.read_csv('train.csv')['PassengerId'].apply(lambda x: x.split('_')[0]).values
y_arr = y.values if hasattr(y, 'values') else y

JOBS = [
    ('LightGBM', lambda: run_lightgbm(X, y, X_test, groups, cat_cols), 'lgbm'),
    ('CatBoost', lambda: run_catboost(X, y, X_test, groups, cat_cols), 'catboost'),
    ('XGBoost', lambda: run_xgboost(X, y, X_test, groups, cat_cols), 'xgb'),
    ('LogReg', lambda: run_lr(X_lin, y, X_test_lin, groups), 'lr'),
    ('KNN', lambda: run_knn(X_lin, y, X_test_lin, groups), 'knn'),
    ('ExtraTrees', lambda: run_extratrees(X_lin, y, X_test_lin, groups), 'extratrees'),
    ('HistGB', lambda: run_histgb(X, y, X_test, groups, cat_cols), 'histgb'),
]

print(f"Data: tree X={X.shape}, linear X={X_lin.shape}, categoricals={len(cat_cols)}")

records = []
for model_name, fn, prefix in JOBS:
    print('\n' + '-' * 72)
    print(f'[{model_name}] training ...')
    train_time, mem_before, mem_peak, mem_delta, result = measure_run(fn)
    # result = (oof, test, infer_time) — 直接用本次训练的 OOF, 而不是磁盘上的权威 npy
    oof, _test_probs, infer_time = result
    metrics = compute_metrics(y_arr, oof)
    records.append({
        'Model': model_name,
        'Prefix': prefix,
        **metrics,
        'TrainTime_sec': round(train_time, 2),
        'TrainTime_min': round(train_time / 60.0, 3),
        'InferTime_sec': round(infer_time, 4),
        'RSS_Before_MB': round(mem_before, 2) if not np.isnan(mem_before) else np.nan,
        'PeakRSS_MB': round(mem_peak, 2) if not np.isnan(mem_peak) else np.nan,
        'PeakRSS_Delta_MB': round(mem_delta, 2) if not np.isnan(mem_delta) else np.nan,
    })
    print(f"[Summary] Acc={metrics['Accuracy']:.5f}, F1={metrics['F1']:.5f}, AUC={metrics['ROC_AUC']:.5f}")
    print(f"[Summary] Train={train_time:.1f}s, Infer={infer_time*1000:.2f}ms, PeakRSS={mem_peak:.1f}MB")

columns = [
    'Model', 'Prefix', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'LogLoss',
    'TrainTime_sec', 'TrainTime_min', 'InferTime_sec', 'RSS_Before_MB', 'PeakRSS_MB', 'PeakRSS_Delta_MB'
]
metrics_df = pd.DataFrame(records)[columns]
metrics_df.to_csv('evaluation_metrics.csv', index=False)
print('\n' + '=' * 72)
print('Aggregated metrics (sorted by Accuracy desc)')
print('=' * 72)
print(metrics_df.sort_values('Accuracy', ascending=False).drop(columns=['Prefix']).to_string(index=False))
print('\nMetrics table saved: evaluation_metrics.csv')
print(f'Run log saved: {LOG_PATH}')
print('Single-model probs saved to: npy_new/  (this run)')
print('Authoritative probs at root *_v70.npy are unchanged (used by ensemble/Level2/Level3 for reproducibility).')
print('Now run level1_ensemble_blend.py for ensemble comparison.')
print('=' * 72)
