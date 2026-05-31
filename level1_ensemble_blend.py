"""
Spaceship Titanic - multi-scale blend comparison (3M ~ 7M, v7.0)
=====================================================
Based on available v70 Single-model OOF/Test probabilities, systematically compare blends of 3/4/5/6/7 models.

Add order (from core to edge, each adding one new diversity):
  3M = LGB + CB + XGB
  4M = 3M + LR
  5M = 4M + KNN
  6M = 5M + ExtraTrees
  7M = 6M + HistGB

For each N output two submissions:
  • ensemble_<N>m_avg.csv   simple equal-weight average, anti-overfit
  • ensemble_<N>m_auto.csv  grid optimal weights, OOF high
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = "submissions"
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.5
REPORT_5M_FIXED_WEIGHTS = {
    "LGB": 0.05,
    "CB": 0.75,
    "XGB": 0.20,
    "LR": 0.00,
    "KNN": 0.00,
}

# ---------- 1. Loading probabilities ----------
PROB_FILES = {
    "LGB": ("lgbm_oof_probs_v70.npy",       "lgbm_test_probs_v70.npy"),
    "CB":  ("catboost_oof_probs_v70.npy",   "catboost_test_probs_v70.npy"),
    "XGB": ("xgb_oof_probs_v70.npy",        "xgb_test_probs_v70.npy"),
    "LR":  ("lr_oof_probs_v70.npy",         "lr_test_probs_v70.npy"),
    "KNN": ("knn_oof_probs_v70.npy",        "knn_test_probs_v70.npy"),
    "ET":  ("extratrees_oof_probs_v70.npy", "extratrees_test_probs_v70.npy"),
    "HGB": ("histgb_oof_probs_v70.npy",     "histgb_test_probs_v70.npy"),
}

oofs, tests = {}, {}
for name, (op, tp) in PROB_FILES.items():
    if os.path.exists(op) and os.path.exists(tp):
        oofs[name], tests[name] = np.load(op), np.load(tp)

test_df = pd.read_csv("test.csv")
y = pd.read_csv("train.csv")["Transported"].astype(int).values

print("Single-model OOF (threshold 0.5):")
for name, oof in oofs.items():
    print(f"   {name:5s}: {accuracy_score(y, (oof > THRESHOLD).astype(int)):.5f}")


# ---------- 2. gridsearched (discrete partition, implemented by itertools) ----------
def grid_search(model_names, step):
    """Enumerate all discrete partitions of sum(w)=1, return (best_weights, best_acc, n_tried)."""
    n = len(model_names)
    n_step = int(round(1.0 / step))
    oof_stack = np.stack([oofs[m] for m in model_names])  # (n, n_samples)

    # Distribute n_step parts into n bins: enumerate using stars-and-bars
    # Equivalent to choosing n-1 separators in n_step + n - 1 positions
    from itertools import combinations
    best_acc, best_w = 0.0, None
    n_tried = 0
    for sep in combinations(range(n_step + n - 1), n - 1):
        # Convert separators to counts per bin
        prev, parts = -1, []
        for s in sep:
            parts.append(s - prev - 1); prev = s
        parts.append(n_step + n - 2 - prev)
        weights = np.array(parts, dtype=float) / n_step
        blend = (weights[:, None] * oof_stack).sum(axis=0)
        acc = accuracy_score(y, (blend > THRESHOLD).astype(int))
        n_tried += 1
        if acc > best_acc:
            best_acc, best_w = acc, weights
    return best_w, best_acc, n_tried


# ---------- 3. Blend for each N ----------
SCHEDULE = [
    ("3m", ["LGB", "CB", "XGB"]),
    ("4m", ["LGB", "CB", "XGB", "LR"]),
    ("5m", ["LGB", "CB", "XGB", "LR", "KNN"]),
    ("6m", ["LGB", "CB", "XGB", "LR", "KNN", "ET"]),
    ("7m", ["LGB", "CB", "XGB", "LR", "KNN", "ET", "HGB"]),
]
STEP_BY_N = {3: 0.01, 4: 0.02, 5: 0.05, 6: 0.05, 7: 0.05}


def save_submission(path, test_blend):
    preds = (test_blend > THRESHOLD).astype(bool)
    pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": preds}) \
      .to_csv(path, index=False)
    return preds


print("\n" + "=" * 72)
print("N-scale blend comparison")
print("=" * 72)

results = []
for tag, names in SCHEDULE:
    if any(m not in oofs for m in names):
        print(f"\n[{tag}] missing models, skip"); continue

    # avg
    avg_oof  = np.mean([oofs[m] for m in names], axis=0)
    avg_test = np.mean([tests[m] for m in names], axis=0)
    avg_acc  = accuracy_score(y, (avg_oof > THRESHOLD).astype(int))
    avg_path = f"{OUT_DIR}/ensemble_{tag}_avg.csv"
    avg_pred = save_submission(avg_path, avg_test)

    # auto
    t0 = time.time()
    if tag == "5m":
        best_w = np.array([REPORT_5M_FIXED_WEIGHTS[m] for m in names], dtype=float)
        fixed_oof = sum(w * oofs[m] for w, m in zip(best_w, names))
        best_acc = accuracy_score(y, (fixed_oof > THRESHOLD).astype(int))
        n_tried = 1
    else:
        best_w, best_acc, n_tried = grid_search(names, STEP_BY_N[len(names)])
    auto_test = sum(w * tests[m] for w, m in zip(best_w, names))
    auto_path = f"{OUT_DIR}/ensemble_{tag}_auto.csv"
    auto_pred = save_submission(auto_path, auto_test)
    weight_str = " ".join(f"{n_}={w:.2f}" for n_, w in zip(names, best_w))

    print(f"\n[{tag}] {' + '.join(names)}")
    print(f"   avg   OOF={avg_acc:.5f}  ({avg_pred.mean()*100:.1f}% True)  → {avg_path}")
    print(f"   auto  OOF={best_acc:.5f}  ({auto_pred.mean()*100:.1f}% True)  → {auto_path}")
    if tag == "5m":
        print(f"      fixed report weights, {time.time()-t0:.1f}s")
    else:
        print(f"      searched {n_tried} combinations, step {STEP_BY_N[len(names)]}, {time.time()-t0:.1f}s")
    print(f"      weights: {weight_str}")

    results.append({"N": len(names), "tag": tag, "avg": avg_acc, "auto": best_acc})


# ---------- 4. Summary ----------
print("\n" + "=" * 72)
print("Summary comparison (5m as baseline)")
print("=" * 72)
base = next((r for r in results if r["tag"] == "5m"), None)
print(f"{'N':<3s} {'tag':<5s} {'avg OOF':<10s} {'auto OOF':<10s} {'Δavg vs 5m':<12s} {'Δauto vs 5m'}")
print("-" * 65)
for r in results:
    da = r["avg"]  - base["avg"]  if base else 0
    du = r["auto"] - base["auto"] if base else 0
    print(f"{r['N']:<3d} {r['tag']:<5s} {r['avg']:.5f}    {r['auto']:.5f}    {da:+.5f}     {du:+.5f}")

print(f"\nAll comparisons generated in {OUT_DIR}/")
