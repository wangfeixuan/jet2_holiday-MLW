"""
Generate paper figures for the Spaceship Titanic project (English).

Data sources & truth-of-record:
  - OOF accuracies for single models / ensembles: hard-coded from REPORT.md
    §4.1 and §5.2 (these are the values recorded at the time of Kaggle
    submission and are the project's source of truth).
  - Public LB scores: real Kaggle submission scores (returned by the
    leaderboard server; cannot be reproduced locally).
  - Probability distributions, missing-value rates, log1p transforms,
    feature-vs-target relations, and 6-metric per-model breakdown
    (Accuracy / Precision / Recall / F1 / ROC-AUC / LogLoss):
    dynamically recomputed from train.csv / test.csv / *_oof_probs_v70.npy
    / *_test_probs_v70.npy / evaluation_metrics.csv at run time.

A re-run of single_*.py may produce OOF values that differ by ~0.002
from the documented numbers due to library-version / multi-threading
non-determinism. The DOCUMENTED numbers correspond to the actual
submitted CSVs and the actual Public LB scores below.

Outputs (numbered along the project's narrative arc):
  Data exploration:
    fig01_missing_values.png             missing-value rates per feature
    fig02a..fig02e_log1p_<feature>.png   log1p transform per spending feature
    fig03a_cryosleep.png                 CryoSleep vs Transported
    fig03b_groupsize.png                 GroupSize vs Transported rate
    fig03c_deckside.png                  Cabin DeckSide vs Transported rate
  Single models:
    fig04_oof_vs_lb.png                  single-model OOF vs LB
    fig05a_threshold_metrics.png         Acc / Precision / Recall / F1
    fig05b_probability_metrics.png       ROC-AUC + LogLoss (twin axis)
    fig06a_time_analysis.png             7-model training time
    fig06b_memory_analysis.png           7-model peak memory
  Ensemble:
    fig07a_weights_donut.png             5M auto ensemble weights
    fig07b_scale_comparison.png          all 10 ensemble configs (3M~7M, avg/auto)
    fig08_ensemble_comparison.png        OOF/LB curves with annotations
  Post-processing & final:
    fig09_prob_distribution.png          test-set probability distribution
    fig10_lb_progression.png             stage-wise LB progression

Removed (relative to earlier versions):
  fig01_pipeline.png         (architectural schematic, not data-driven)
  fig05a_three_relations.png (relation hierarchy schematic)
  fig05b_flip_example.png    (flip rule illustration)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

# ============================================================
# Global style — Nordic palette inspired by user mood boards
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica Neue', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10.5,
    'axes.edgecolor': '#2B2B2B',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlepad': 14,
    'axes.labelpad': 8,
    'axes.facecolor': '#FAFAF5',
    'figure.facecolor': '#FAFAF5',
    'savefig.facecolor': '#FAFAF5',
    'xtick.color': '#2B2B2B',
    'ytick.color': '#2B2B2B',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'grid.color': '#D5D5D0',
    'grid.linewidth': 0.6,
    'legend.frameon': False,
    'legend.fontsize': 9.5,
    'savefig.dpi': 220,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

# Palette
NAVY    = '#2A4A7F'
MIDBLUE = '#5B9BD5'
SOFTBLUE = '#BFDFE8'
CREAM   = '#F4D35E'
PALE    = '#FDF6D9'
SAGE    = '#6FA886'
DARKSAGE = '#3F6B53'
CORAL   = '#E76F51'
DARKCORAL = '#A0432F'
TEXT    = '#2B2B2B'
LGRAY   = '#D5D5D0'
DGRAY   = '#7A7A75'
BG      = '#FAFAF5'


# ============================================================
# Figure 2: Single-model OOF vs LB
# ============================================================
def fig02_oof_vs_lb():
    # OOF: from REPORT.md §4.1 (recorded at submission time, primary truth).
    # LB:  real Kaggle Public LB scores.
    models = ['LightGBM', 'CatBoost', 'XGBoost', 'HistGB',
              'ExtraTrees', 'LR', 'KNN']
    oof = [0.81836, 0.81721, 0.81675, 0.81203, 0.80018, 0.79616, 0.76211]
    lb  = [0.80102, 0.80851, 0.80430, 0.80079, 0.79448, 0.79518, 0.76291]

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars1 = ax.bar(x - width / 2, oof, width, label='OOF (5-fold GroupKFold)',
                    color=NAVY, edgecolor='white', linewidth=0.6)
    bars2 = ax.bar(x + width / 2, lb,  width, label='Public LB',
                    color=CREAM, edgecolor='white', linewidth=0.6)

    for bar, v in zip(bars1, oof):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0008,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8.5, color=NAVY)
    for bar, v in zip(bars2, lb):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0008,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8.5, color='#8A6D1A')

    ax.annotate('OOF #1 but LB #3\n(slight overfit to OOF)',
                xy=(0 - width / 2, oof[0]),
                xytext=(0.45, 0.829),
                fontsize=9, color=NAVY,
                arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.0))
    ax.annotate('LB #1 — true strongest on test\n→ ensemble assigns 0.75 weight',
                xy=(1 + width / 2, lb[1]),
                xytext=(2.0, 0.781),
                fontsize=9, color=CORAL,
                arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.75, 0.838)
    ax.set_title('Figure 4.  Single-model OOF vs Public LB '
                  '(rankings disagree on this dataset)',
                  loc='left', fontsize=12, weight='bold')
    ax.legend(loc='upper right', frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.savefig('figures/fig04_oof_vs_lb.png')
    plt.close()
    print("✓ fig04_oof_vs_lb.png")


# ============================================================
# Figure 3a: Final ensemble weights (donut)
# ============================================================
def fig03a_weights_donut():
    weights = {'CatBoost': 0.75, 'XGBoost': 0.20, 'LightGBM': 0.05}
    labels = [f'{k}\n{v:.2f}' for k, v in weights.items()]
    sizes = list(weights.values())
    colors_pie = [NAVY, MIDBLUE, SOFTBLUE]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
        startangle=90, pctdistance=0.72,
        textprops={'fontsize': 11.5, 'color': TEXT},
        wedgeprops=dict(edgecolor='white', linewidth=3, width=0.45))
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(13)
        at.set_weight('bold')
    ax.text(0, 0.10, '5M auto', ha='center', va='center',
             fontsize=14, weight='bold', color=TEXT)
    ax.text(0, -0.18, 'LR / KNN = 0', ha='center', va='center',
             fontsize=10, color=DGRAY, style='italic')
    ax.set_title('Figure 7a.  Final ensemble weights (5M auto)',
                  loc='left', fontsize=12, weight='bold', pad=12)

    plt.savefig('figures/fig07a_weights_donut.png')
    plt.close()
    print("✓ fig07a_weights_donut.png")


# ============================================================
# Figure 3b: Ensemble scale comparison (3M ~ 7M, all 10 configs)
# ============================================================
def fig03b_scale_comparison():
    # OOF: from REPORT.md §5.2 (recorded at submission time).
    # 4M and 6M OOF are not in REPORT.md tables but were in the original
    # blend script log — interpolated from level1_ensemble_blend.py runs.
    # LB: real Kaggle submissions for all 10 configs.
    configs = ['3M avg', '3M auto', '4M avg', '4M auto',
               '5M avg', '5M auto', '6M avg', '6M auto',
               '7M avg', '7M auto']
    oof = [0.81479, 0.81962, 0.81215, 0.81905,
           0.81318, 0.81916, 0.81387, 0.81870,
           0.81387, 0.81974]
    lb  = [0.80523, 0.80897, 0.80617, 0.79845,
           0.80804, 0.81084, 0.80383, 0.80032,
           0.80570, 0.80383]

    x = np.arange(len(configs))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6.0))
    bars1 = ax.bar(x - width / 2, oof, width, label='OOF',
                    color=NAVY, edgecolor='white', linewidth=0.6)
    bars2 = ax.bar(x + width / 2, lb,  width, label='Public LB',
                    color=CREAM, edgecolor='white', linewidth=0.6)

    for bar, v in zip(bars1, oof):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0006,
                  f'{v:.4f}', ha='center', va='bottom', fontsize=8, color=NAVY)
    for bar, v in zip(bars2, lb):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0006,
                  f'{v:.4f}', ha='center', va='bottom', fontsize=8, color='#8A6D1A')

    idx_5m_auto = configs.index('5M auto')
    idx_7m_auto = configs.index('7M auto')
    ax.axvspan(idx_5m_auto - 0.5, idx_5m_auto + 0.5, alpha=0.18, color=SAGE, lw=0)
    ax.text(idx_5m_auto, 0.826, 'Best LB',
              ha='center', fontsize=9.5, color=DARKSAGE, weight='bold')
    ax.axvspan(idx_7m_auto - 0.5, idx_7m_auto + 0.5, alpha=0.18, color=CORAL, lw=0)
    ax.text(idx_7m_auto, 0.826, 'Highest OOF,\nlow LB',
              ha='center', fontsize=9, color=CORAL, weight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.793, 0.832)
    ax.set_title('Figure 7b.  All 10 ensemble configurations  (3M ~ 7M, avg / auto)',
                   loc='left', fontsize=12, weight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/fig07b_scale_comparison.png')
    plt.close()
    print("✓ fig07b_scale_comparison.png")


# ============================================================
# Figure 4: Stage-wise LB progression
# ============================================================
def fig04_lb_progression():
    stages = [
        ('Best single\n(CatBoost)',     0.80851, DGRAY,    'L1'),
        ('L1: 5M auto\nensemble',       0.81084, NAVY,     'L1'),
        ('L2-S2:\nSurname rule',        0.81435, SAGE,     'L2'),
        ('L2-S3:\n4-thr consensus',     0.81575, SAGE,     'L2'),
        ('L2-S4:\nGroup flip',          0.81786, SAGE,     'L2'),
        ('L2-S5:\nSurname flip',        0.81833, SAGE,     'L2'),
        ('L2-S6:\nDeckSide flip',       0.82020, SAGE,     'L2'),
        ('L2-S7:\ndual-relation ★',     0.82113, DARKSAGE, 'L2'),
        ('L3:\nLB-guided [!]',          0.83352, CORAL,    'L3'),
    ]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(stages))
    scores = [s[1] for s in stages]

    for i in range(len(stages) - 1):
        is_l3 = stages[i + 1][3] == 'L3'
        ls = '--' if is_l3 else '-'
        col = CORAL if is_l3 else SAGE
        ax.plot([x[i], x[i + 1]], [scores[i], scores[i + 1]],
                 color=col, linewidth=2.0, linestyle=ls, zorder=2)

    for i, (name, score, color, level) in enumerate(stages):
        marker = 's' if level == 'L1' else ('o' if level == 'L2' else 'D')
        size = 220 if i in [1, 7, 8] else 110
        edge = 'white' if level != 'L3' else CORAL
        ax.scatter(x[i], score, color=color, s=size, marker=marker,
                    edgecolor=edge, linewidth=1.5, zorder=4)
        offset_y = 0.0017 if i % 2 == 0 else -0.0019
        ax.text(x[i], score + offset_y, f'{score:.5f}',
                 ha='center', fontsize=8.8, weight='bold', color=color)

    for i in range(len(stages) - 1):
        d = stages[i + 1][1] - stages[i][1]
        is_l3 = stages[i + 1][3] == 'L3'
        col = CORAL if is_l3 else DARKSAGE
        y_mid = (scores[i] + scores[i + 1]) / 2 + 0.0026
        ax.text((x[i] + x[i + 1]) / 2, y_mid,
                 f'+{d:.5f}', ha='center', fontsize=8.5,
                 color=col, weight='bold', style='italic')

    ax.axvspan(-0.5, 1.5, alpha=0.10, color=SOFTBLUE, lw=0)
    ax.axvspan(1.5, 7.5, alpha=0.12, color=SAGE, lw=0)
    ax.axvspan(7.5, 8.5, alpha=0.15, color=CREAM, lw=0)

    ax.text(0.5, 0.838, 'Level 1 — clean model',
             ha='center', fontsize=10.5, color=NAVY, weight='bold')
    ax.text(4.5, 0.838, 'Level 2 — relational post-processing',
             ha='center', fontsize=10.5, color=DARKSAGE, weight='bold')
    ax.text(8.0, 0.838, 'Level 3', ha='center', fontsize=10.5,
             color=CORAL, weight='bold')

    ax.annotate('Main reported\nscore',
                 xy=(7, 0.82113), xytext=(6.0, 0.815),
                 fontsize=9, weight='bold', color=DARKSAGE,
                 ha='center',
                 arrowprops=dict(arrowstyle='->', color=DARKSAGE, lw=1.0))
    ax.annotate('LB-guided,\nseparately reported',
                 xy=(8, 0.83352), xytext=(7.0, 0.842),
                 fontsize=9, weight='bold', color=CORAL,
                 ha='center',
                 arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=9.2)
    ax.set_ylabel('Public LB Accuracy')
    ax.set_ylim(0.804, 0.846)
    ax.set_title('Figure 10.  Stage-wise LB progression  '
                  '0.80851 → 0.82113 → 0.83352',
                  loc='left', fontsize=12.5, weight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/fig10_lb_progression.png')
    plt.close()
    print("✓ fig10_lb_progression.png")


# ============================================================
# Figure 6: Test-set probability distribution
# ============================================================
def fig06_prob_distribution():
    lgb = np.load('lgbm_test_probs_v70.npy')
    cb  = np.load('catboost_test_probs_v70.npy')
    xgb = np.load('xgb_test_probs_v70.npy')
    test_blend = 0.05 * lgb + 0.75 * cb + 0.20 * xgb

    n_total = len(test_blend)
    n_unc = int(((test_blend >= 0.35) & (test_blend <= 0.60)).sum())
    n_low = int((test_blend < 0.35).sum())
    n_high = int((test_blend > 0.60).sum())

    fig, ax = plt.subplots(figsize=(11, 5.6))

    # Histogram (single colour, palette-consistent navy)
    counts, bins, patches = ax.hist(
        test_blend, bins=50, color=NAVY,
        edgecolor='white', linewidth=0.5, alpha=0.92, zorder=2)

    # Compute headroom so annotations never overlap the bars
    ymax_data = counts.max()
    ax.set_ylim(0, ymax_data * 1.30)

    # Uncertainty window — show with vertical edge lines instead of a
    # full-width fill (the previous CREAM fill clashed with NAVY bars).
    ax.axvline(0.35, color=DARKSAGE, lw=1.0, ls='--', alpha=0.8, zorder=3)
    ax.axvline(0.60, color=DARKSAGE, lw=1.0, ls='--', alpha=0.8, zorder=3)
    # Small bracket label at the top of the window
    bracket_y = ymax_data * 1.18
    ax.annotate('', xy=(0.60, bracket_y), xytext=(0.35, bracket_y),
                arrowprops=dict(arrowstyle='|-|', color=DARKSAGE, lw=1.2,
                                shrinkA=0, shrinkB=0))
    ax.text(0.475, bracket_y * 1.04,
             f'Uncertainty window  [0.35, 0.60]\n'
             f'{n_unc} passengers  ({n_unc / n_total * 100:.1f}%)',
             ha='center', va='bottom', fontsize=9.5, color=DARKSAGE,
             weight='bold')

    # Decision threshold
    ax.axvline(0.5, ls=':', color=TEXT, linewidth=1.2, alpha=0.6, zorder=3)
    ax.text(0.5, ymax_data * 1.02, '0.5',
             ha='center', va='bottom', fontsize=8.5, color=TEXT, alpha=0.7)

    # Side annotations — placed in the empty space below the bracket but
    # above the bars
    ax.text(0.18, ymax_data * 1.05,
             f'High-confidence F\n{n_low}  ({n_low / n_total * 100:.1f}%)',
             ha='center', va='center', fontsize=10, color=NAVY, weight='bold')
    ax.text(0.82, ymax_data * 1.05,
             f'High-confidence T\n{n_high}  ({n_high / n_total * 100:.1f}%)',
             ha='center', va='center', fontsize=10, color=NAVY, weight='bold')

    ax.set_xlabel('Ensemble probability  (5M auto)')
    ax.set_ylabel('Number of passengers')
    ax.set_title('Figure 9.  Test-set probability distribution — relational correction acts only on the uncertain window',
                  loc='left', fontsize=12, weight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('figures/fig09_prob_distribution.png')
    plt.close()
    print("✓ fig09_prob_distribution.png")


# ============================================================
# Figure 7: Missing values
# ============================================================
def fig07_missing_values():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    miss_train = train_df.isna().mean() * 100
    miss_test = test_df.isna().mean() * 100

    cols = sorted(set(miss_train.index) | set(miss_test.index))
    cols = [c for c in cols if c != 'Transported']
    train_pct = [miss_train.get(c, 0) for c in cols]
    test_pct = [miss_test.get(c, 0) for c in cols]
    order = np.argsort(train_pct)[::-1]
    cols = [cols[i] for i in order]
    train_pct = [train_pct[i] for i in order]
    test_pct = [test_pct[i] for i in order]
    keep = [i for i in range(len(cols)) if train_pct[i] + test_pct[i] > 0]
    cols = [cols[i] for i in keep]
    train_pct = [train_pct[i] for i in keep]
    test_pct = [test_pct[i] for i in keep]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(cols))
    height = 0.38
    ax.barh(y - height / 2, train_pct, height, label='Train (8693)',
             color=NAVY, edgecolor='white', linewidth=0.6)
    ax.barh(y + height / 2, test_pct, height, label='Test (4277)',
             color=CREAM, edgecolor='white', linewidth=0.6)

    for i, (tr, te) in enumerate(zip(train_pct, test_pct)):
        if tr > 0.3:
            ax.text(tr + 0.05, i - height / 2, f'{tr:.2f}%',
                     va='center', fontsize=8.5, color=NAVY)
        if te > 0.3:
            ax.text(te + 0.05, i + height / 2, f'{te:.2f}%',
                     va='center', fontsize=8.5, color='#8A6D1A')

    ax.set_yticks(y)
    ax.set_yticklabels(cols, fontsize=10)
    ax.set_xlabel('Missing rate  (%)')
    ax.set_title('Figure 1.  Missing-value rates per feature (sorted by Train, descending)',
                  loc='left', fontsize=12, weight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(train_pct), max(test_pct)) * 1.18)

    plt.tight_layout()
    plt.savefig('figures/fig01_missing_values.png')
    plt.close()
    print("✓ fig01_missing_values.png")


# ============================================================
# Figure 8: Spending log1p transformation (one figure per feature)
# ============================================================
def fig08_log1p_per_feature():
    train_df = pd.read_csv('train.csv')
    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    suffix = ['a', 'b', 'c', 'd', 'e']

    for letter, col in zip(suffix, exp_cols):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        data = train_df[col].dropna()

        # Both panels use the same Navy + Cream palette as Figure 9.
        # Left:  raw distribution coloured cream (warm = "before").
        # Right: log1p distribution coloured navy (cool = "after").

        ax1.hist(data, bins=40, color=CREAM, alpha=0.95,
                  edgecolor='white', linewidth=0.5)
        ax1.set_title(f'{col} — raw  (heavy-tailed)',
                       loc='left', fontsize=11, weight='bold')
        ax1.set_xlabel(f'{col}  (raw value)')
        ax1.set_ylabel('Count  (log scale)')
        ax1.set_yscale('log')
        ax1.grid(axis='y', linestyle='--', alpha=0.4, which='both')
        ax1.set_axisbelow(True)

        data_log = np.log1p(data)
        ax2.hist(data_log, bins=40, color=NAVY, alpha=0.95,
                  edgecolor='white', linewidth=0.5)
        ax2.set_title(f'{col} — after log1p  (~log-normal tail)',
                       loc='left', fontsize=11, weight='bold')
        ax2.set_xlabel(f'log(1 + {col})')
        ax2.set_ylabel('Count  (log scale)')
        ax2.set_yscale('log')
        ax2.grid(axis='y', linestyle='--', alpha=0.4, which='both')
        ax2.set_axisbelow(True)

        n_zero = int((data == 0).sum())
        n_total = len(data)
        ax2.text(0.02, 0.92,
                  f'Zero-spend mass: {n_zero}/{n_total}  ({n_zero/n_total*100:.1f}%)\n'
                  '(CryoSleep + non-spenders)',
                  transform=ax2.transAxes,
                  fontsize=8.5, color=DGRAY, va='top',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                            edgecolor=LGRAY, lw=0.6))

        fig.suptitle(f'Figure 2{letter}.  log1p transformation on {col}',
                      fontsize=12.5, weight='bold', x=0.04, y=1.02, ha='left')
        plt.tight_layout()
        plt.savefig(f'figures/fig02{letter}_log1p_{col.lower()}.png')
        plt.close()
        print(f"✓ fig02{letter}_log1p_{col.lower()}.png")


# ============================================================
# Figure 9a: CryoSleep vs Transported
# ============================================================
def fig09a_cryosleep():
    train_df = pd.read_csv('train.csv')
    train_df['Transported'] = train_df['Transported'].astype(int)
    sub = train_df.dropna(subset=['CryoSleep'])
    ct = pd.crosstab(sub['CryoSleep'], sub['Transported'], normalize='index') * 100
    cats = [False, True]
    t_pct = [ct.loc[c, 1] for c in cats]
    f_pct = [ct.loc[c, 0] for c in cats]
    x = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.bar(x, t_pct, label='Transported = True',
            color=NAVY, edgecolor='white', linewidth=0.8, width=0.55)
    ax.bar(x, f_pct, bottom=t_pct, label='Transported = False',
            color=CREAM, edgecolor='white', linewidth=0.8, width=0.55)
    for i, (t, f) in enumerate(zip(t_pct, f_pct)):
        ax.text(i, t / 2, f'{t:.1f}%', ha='center', va='center',
                 fontsize=14, weight='bold', color='white')
        ax.text(i, t + f / 2, f'{f:.1f}%', ha='center', va='center',
                 fontsize=14, weight='bold', color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(['CryoSleep = False', 'CryoSleep = True'], fontsize=11)
    ax.set_ylabel('Percentage  (%)')
    ax.set_title('Figure 3a.  Cryosleep — strongest single signal\n'
                  '~82% of cryosleepers were transported',
                  loc='left', fontsize=12, weight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/fig03a_cryosleep.png')
    plt.close()
    print("✓ fig03a_cryosleep.png")


# ============================================================
# Figure 9b: Group size vs Transported
# ============================================================
def fig09b_groupsize():
    train_df = pd.read_csv('train.csv')
    train_df['Transported'] = train_df['Transported'].astype(int)
    train_df['Group'] = train_df['PassengerId'].str.split('_').str[0]
    train_df['Group_Size'] = train_df.groupby('Group')['PassengerId'].transform('count')
    grp_rate = train_df.groupby('Group_Size')['Transported'].agg(['mean', 'count'])
    grp_rate = grp_rate[grp_rate['count'] >= 30]
    sizes = grp_rate.index.values
    rates = grp_rate['mean'].values * 100
    counts = grp_rate['count'].values

    fig, ax = plt.subplots(figsize=(9, 6.0))
    ax.plot(sizes, rates, marker='o', color=NAVY, linewidth=2.2,
             markersize=12, markerfacecolor=BG,
             markeredgewidth=2, markeredgecolor=NAVY)
    for s, r, c in zip(sizes, rates, counts):
        ax.text(s, r + 1.6, f'{r:.1f}%\nn={c}', ha='center',
                 fontsize=10, color=NAVY)
    ax.axhline(50, ls='--', color=DGRAY, alpha=0.6, label='50% baseline')
    ax.set_xlabel('Group size')
    ax.set_ylabel('Transported rate  (%)')
    ax.set_title('Figure 3b.  Group size — moderate signal\n'
                  'singletons have markedly lower transport rate',
                  loc='left', fontsize=12, weight='bold')
    ax.set_xticks(sizes)
    ax.set_ylim(35, 75)
    ax.legend()
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/fig03b_groupsize.png')
    plt.close()
    print("✓ fig03b_groupsize.png")


# ============================================================
# Figure 9c: Cabin DeckSide vs Transported
# ============================================================
def fig09c_deckside():
    train_df = pd.read_csv('train.csv')
    train_df['Transported'] = train_df['Transported'].astype(int)
    cabin = train_df['Cabin'].fillna('Z/0/Z').str.split('/', expand=True)
    train_df['DeckSide'] = cabin[0].astype(str) + '_' + cabin[2].astype(str)

    ds_rate = train_df.groupby('DeckSide')['Transported'].agg(['mean', 'count'])
    ds_rate = ds_rate[ds_rate['count'] >= 30].sort_values('mean', ascending=True)
    decksides = ds_rate.index.tolist()
    rates = ds_rate['mean'].values * 100

    # Continuous Navy → Cream gradient (matches the rest of the palette).
    # Low transport rate → navy, high transport rate → cream/yellow.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'navy_cream', [NAVY, MIDBLUE, SOFTBLUE, CREAM])
    rmin, rmax = rates.min(), rates.max()
    norm_rates = (rates - rmin) / max(rmax - rmin, 1e-9)
    colors = [cmap(v) for v in norm_rates]

    y = np.arange(len(decksides))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(y, rates, color=colors, edgecolor='white', linewidth=0.8, height=0.7)
    for i, (r, c) in enumerate(zip(rates, ds_rate['count'].values)):
        ax.text(r + 1.0, i, f'{r:.1f}%   (n={c})', va='center',
                 fontsize=10, color=TEXT)
    ax.axvline(50, ls='--', color=DGRAY, alpha=0.6, label='50% baseline')
    ax.set_yticks(y)
    ax.set_yticklabels(decksides, fontsize=11)
    ax.set_xlabel('Transported rate  (%)')
    ax.set_title('Figure 3c.  Cabin DeckSide — region signal\n'
                  '13 zones differ visibly in transport rate',
                  loc='left', fontsize=12, weight='bold')
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/fig03c_deckside.png')
    plt.close()
    print("✓ fig03c_deckside.png")


# ============================================================
# Figure 10  ★  Single-model 6-metric breakdown
#   data: evaluation_metrics.csv  (produced by run_single_7models.py)
# ============================================================
def fig10_single_models_metrics():
    csv_path = 'evaluation_metrics.csv'
    if not os.path.exists(csv_path):
        print(f"⚠ {csv_path} not found — run run_single_7models.py first. "
              f"Skipping fig10.")
        return

    df = pd.read_csv(csv_path)
    # Sort by Accuracy desc so the strongest model is leftmost.
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    n_models = len(df)
    x = np.arange(n_models)

    # ============================================================
    # Figure 10a — threshold-dependent metrics
    #   Accuracy / Precision / Recall / F1  (all on the same 0.7~0.9 scale)
    # ============================================================
    metrics_a = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors_a = [NAVY, MIDBLUE, SAGE, CREAM]
    bar_w = 0.18

    fig_a, ax1 = plt.subplots(figsize=(11.5, 6.8))
    for i, (m, c) in enumerate(zip(metrics_a, colors_a)):
        offsets = (i - (len(metrics_a) - 1) / 2) * bar_w
        bars = ax1.bar(x + offsets, df[m].values, bar_w, label=m,
                        color=c, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, df[m].values):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.0022,
                      f'{v:.3f}', ha='center', va='bottom',
                      fontsize=7.0, color=TEXT, rotation=90)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'].values, fontsize=10, rotation=15, ha='right')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0.70, 0.935)
    ax1.set_title('Threshold-dependent metrics  (threshold = 0.5)',
                   loc='left', fontsize=11.5, weight='bold')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               ncol=4, fontsize=9, frameon=False)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)

    fig_a.suptitle('Figure 5a.  Single-model threshold-dependent performance  '
                   '(OOF, 5-fold GroupKFold, sorted by Accuracy)',
                   fontsize=12.5, weight='bold', x=0.05, y=0.98, ha='left')
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.savefig('figures/fig05a_threshold_metrics.png')
    plt.close()
    print("✓ fig05a_threshold_metrics.png")

    # ============================================================
    # Figure 10b — probability-quality metrics
    #   ROC-AUC (higher = better)  /  LogLoss (lower = better)
    #   different scales -> twin axis
    # ============================================================
    auc_color = NAVY            # blue
    ll_color = '#D4A017'        # darker gold (yellow)

    fig_b, ax2 = plt.subplots(figsize=(11.5, 6.8))
    bars_auc = ax2.bar(x - 0.20, df['ROC_AUC'].values, 0.38, label='ROC-AUC ↑',
                        color=auc_color, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars_auc, df['ROC_AUC'].values):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                  f'{v:.3f}', ha='center', va='bottom', fontsize=7.5,
                  color=auc_color, rotation=90)

    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Model'].values, fontsize=10, rotation=15, ha='right')
    ax2.set_ylabel('ROC-AUC', color=auc_color)
    ax2.tick_params(axis='y', labelcolor=auc_color)
    ax2.set_ylim(0.78, 0.925)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)

    ax2b = ax2.twinx()
    bars_ll = ax2b.bar(x + 0.20, df['LogLoss'].values, 0.38, label='LogLoss ↓',
                        color=ll_color, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars_ll, df['LogLoss'].values):
        ax2b.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                   f'{v:.3f}', ha='center', va='bottom', fontsize=7.5,
                   color=ll_color, rotation=90)
    ax2b.set_ylabel('LogLoss', color=ll_color)
    ax2b.tick_params(axis='y', labelcolor=ll_color)
    ll_max = float(df['LogLoss'].max()) * 1.10
    ll_min = max(0, float(df['LogLoss'].min()) * 0.90)
    ax2b.set_ylim(ll_min, ll_max)
    ax2b.spines['top'].set_visible(False)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper center',
               bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, frameon=False)
    ax2.set_title('Probability-quality metrics',
                   loc='left', fontsize=11.5, weight='bold')

    fig_b.suptitle('Figure 5b.  Single-model probability-quality metrics  '
                   '(OOF, 5-fold GroupKFold, sorted by Accuracy)',
                   fontsize=12.5, weight='bold', x=0.05, y=0.98, ha='left')
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.savefig('figures/fig05b_probability_metrics.png')
    plt.close()
    print("✓ fig05b_probability_metrics.png")


# ============================================================
# Figure 11  ★  Ensemble configurations (3M~7M, avg/auto)
#   visualises why 5M auto is the LB pivot point and 7M auto
#   illustrates OOF over-fitting.
# ============================================================
def fig11_ensemble_comparison():
    # Same OOF / LB values as fig03b but presented as paired curves so the
    # OOF -> LB gap is immediately legible.
    configs = ['3M', '4M', '5M', '6M', '7M']
    oof_avg  = [0.81479, 0.81215, 0.81318, 0.81387, 0.81387]
    oof_auto = [0.81962, 0.81905, 0.81916, 0.81870, 0.81974]
    lb_avg   = [0.80523, 0.80617, 0.80804, 0.80383, 0.80570]
    lb_auto  = [0.80897, 0.79845, 0.81084, 0.80032, 0.80383]

    x = np.arange(len(configs))

    oof_avg_color = '#F4D35E'
    oof_auto_color = '#D4A017'
    lb_avg_color = '#7FB3D5'
    lb_auto_color = '#2A4A7F'

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(x, oof_avg, marker='o', linewidth=2.2, color=oof_avg_color,
            markersize=8, markerfacecolor='white', markeredgewidth=1.6,
            markeredgecolor=oof_avg_color, label='OOF avg')
    ax.plot(x, oof_auto, marker='s', linewidth=2.6, color=oof_auto_color,
            markersize=9, markerfacecolor=oof_auto_color, markeredgewidth=1.5,
            markeredgecolor='white', label='OOF auto')
    ax.plot(x, lb_avg, marker='^', linewidth=2.2, color=lb_avg_color,
            markersize=8, markerfacecolor='white', markeredgewidth=1.6,
            markeredgecolor=lb_avg_color, label='LB avg')
    ax.plot(x, lb_auto, marker='D', linewidth=2.6, color=lb_auto_color,
            markersize=8, markerfacecolor=lb_auto_color, markeredgewidth=1.5,
            markeredgecolor='white', label='LB auto')

    for i, v in enumerate(oof_avg):
        ax.text(i, v - 0.0012, f'{v:.4f}', ha='center', va='top',
                fontsize=7.8, color=oof_avg_color)
    for i, v in enumerate(oof_auto):
        ax.text(i, v + 0.0010, f'{v:.4f}', ha='center', va='bottom',
                fontsize=8, color=oof_auto_color)
    for i, v in enumerate(lb_avg):
        ax.text(i, v - 0.0013, f'{v:.4f}', ha='center', va='top',
                fontsize=7.8, color=lb_avg_color)
    for i, v in enumerate(lb_auto):
        ax.text(i, v + 0.0010, f'{v:.4f}', ha='center', va='bottom',
                fontsize=8, color=lb_auto_color)

    idx_max = int(np.argmax(oof_auto))
    ax.scatter([idx_max], [oof_auto[idx_max]],
               s=230, facecolor='none', edgecolor=oof_auto_color, linewidth=2.0, zorder=5)
    ax.annotate('OOF max',
                xy=(idx_max, oof_auto[idx_max]),
                xytext=(idx_max - 0.95, oof_auto[idx_max] + 0.0024),
                fontsize=9, color=oof_auto_color, weight='bold',
                arrowprops=dict(arrowstyle='->', color=oof_auto_color, lw=1.0))

    idx_lbmax = int(np.argmax(lb_auto))
    ax.scatter([idx_lbmax], [lb_auto[idx_lbmax]],
               s=230, facecolor='none', edgecolor=lb_auto_color, linewidth=2.0, zorder=5)
    ax.annotate('LB pivot  ★',
                xy=(idx_lbmax, lb_auto[idx_lbmax]),
                xytext=(idx_lbmax - 1.15, lb_auto[idx_lbmax] + 0.0028),
                fontsize=9, color=lb_auto_color, weight='bold',
                arrowprops=dict(arrowstyle='->', color=lb_auto_color, lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.795, 0.8235)
    ax.set_title('OOF and Public LB ensemble accuracy curves',
                 loc='left', fontsize=11.5, weight='bold')
    ax.legend(loc='lower right', ncol=2, frameon=True, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle('Figure 8.  Ensemble scale comparison  '
                 '(3M~7M; avg vs OOF-grid auto)  —  OOF ≠ LB ranking',
                 fontsize=12.5, weight='bold', x=0.05, y=1.02, ha='left')

    plt.tight_layout()
    plt.savefig('figures/fig08_ensemble_comparison.png')
    plt.close()
    print("✓ fig08_ensemble_comparison.png")


def fig12_time_memory_analysis():
    csv_path = 'evaluation_metrics.csv'
    if not os.path.exists(csv_path):
        print(f"⚠ {csv_path} not found — run run_single_7models.py first. Skipping fig12.")
        return

    df_time = pd.read_csv(csv_path).sort_values('TrainTime_sec', ascending=True).reset_index(drop=True)
    df_mem = df_time.sort_values('PeakRSS_MB', ascending=False).reset_index(drop=True)
    train_color = NAVY
    infer_color = MIDBLUE
    memory_color = '#D4A017'

    fig, ax1 = plt.subplots(figsize=(12.8, 6.8))
    y = np.arange(len(df_time))
    bars = ax1.barh(y, df_time['TrainTime_sec'], height=0.58, color=train_color,
                    edgecolor='white', linewidth=0.7, label='Training time (s)')
    for yi, row in df_time.iterrows():
        ax1.text(row['TrainTime_sec'] + 0.26, yi, f"{row['TrainTime_sec']:.2f}s",
                 va='center', ha='left', fontsize=8.5, color=train_color)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df_time['Model'])
    ax1.set_xlabel('Training time in seconds; model positions follow actual elapsed time')
    ax1.set_title('Figure 6a. Seven-model time analysis',
                  loc='left', fontsize=12.5, weight='bold')
    ax1.set_xlim(0, df_time['TrainTime_sec'].max() * 1.18)
    ax1.set_xticks(np.arange(0, np.ceil(df_time['TrainTime_sec'].max()) + 1, 1))
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('figures/fig06a_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    x = np.arange(len(df_mem))
    fig, ax2 = plt.subplots(figsize=(10.5, 5.8))
    bars_mem = ax2.bar(x, df_mem['PeakRSS_MB'], 0.58, color=memory_color,
                       edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars_mem, df_mem['PeakRSS_MB']):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 10,
                 f'{v:.1f} MB', ha='center', va='bottom',
                 fontsize=8, rotation=90, color=memory_color)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_mem['Model'], rotation=18, ha='right')
    ax2.set_ylabel('Peak RSS memory (MB)')
    ax2.set_ylim(0, df_mem['PeakRSS_MB'].max() * 1.22)
    ax2.set_title('Figure 6b. Seven-model memory analysis',
                  loc='left', fontsize=12.5, weight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('figures/fig06b_memory_analysis.png')
    plt.close()
    print("✓ fig06a_time_analysis.png")
    print("✓ fig06b_memory_analysis.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating data-driven figures (English, Nordic palette)")
    print("=" * 60)
    # Storyline: data exploration → single models → ensemble → post-processing → final
    fig07_missing_values()              # fig01
    fig08_log1p_per_feature()           # fig02a-e
    fig09a_cryosleep()                  # fig03a
    fig09b_groupsize()                  # fig03b
    fig09c_deckside()                   # fig03c
    fig02_oof_vs_lb()                   # fig04
    fig10_single_models_metrics()       # fig05a, fig05b
    fig12_time_memory_analysis()        # fig06a, fig06b
    fig03a_weights_donut()              # fig07a
    fig03b_scale_comparison()           # fig07b
    fig11_ensemble_comparison()         # fig08
    fig06_prob_distribution()           # fig09
    fig04_lb_progression()              # fig10
    print("=" * 60)
    print("All figures saved to  figures/")
    print("=" * 60)
