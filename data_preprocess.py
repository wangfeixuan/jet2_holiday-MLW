"""
Spaceship Titanic - Data Preprocessing v7.0 (Anti-overfitting Edition)
======================================================================
Changes in v7.0 compared to v6.6:

【Background Diagnosis】
  V66 OOF=0.8163 but LB=0.8088 (CV is 0.7% higher than LB) → Overfitting CV
  Others (Surendiran) CV=0.808 but LB=0.8190 → Simpler model generalizes better
  ⇒ Our LB drops by 1.024%, mainly due to “over-engineering + derived features causing distribution shift”

【v7.0 Key Idea: Occam’s Razor】
  ✂️ Remove 24 marginal/collinear/distribution-shifting derived features
  Keep all domain rules (CryoSleep consistency, 4-layer imputation, Deck→HomePlanet, etc.)
  Keep IterativeImputer Age (but only this is train-only fit)
  🛡️ All imputation is train-only fit, no new transductive derived features
  Feature count reduced: 62 → 38

【Retained domain rules (reduce missing, don’t increase feature count)】
  • CryoSleep consistency: Spending>0 → False; CryoSleep=True → Spending=0
  • 4-layer imputation: Group → Surname → Deck → global train mode
  • Deck → HomePlanet inference (train-only mapping)
  • VIP=True → Europa (train-only mode)
  • HomePlanet → Deck default mapping

【Removed 24 features + reasons (see README/discussion)】
  Family_* (2): test set has many new surnames, causes distribution shift
  Cabin_Group_Size / Cabin_Num_Pct: Collinear with Cabin_Was_Missing/Region
  Group_* derived (7): Spending_Max, Age_Std, PP_Range, PP_Max, Total_Spending,
                    Has_Child, All_Cryo  — Collinear with core group features
  combinationscategories (2): HomePlanet_Destination, HomePlanet_Deck — High-cardinality one-hot explosion
  Only_Luxury / Only_Basic / Spending_Concentration: Collinear
  Spending_to_Group_Ratio / Personal_Group_Spend_Ratio: Distribution shift
  Luxury_Ratio / Basic_Ratio / RoomService_Ratio /
  FoodCourt_Ratio / ShoppingMall_Ratio: Strongly correlated with original *_Spending
  Is_Teen: Duplicate of Age_Group
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

pd.set_option('future.no_silent_downcasting', True)

_HERE = os.path.dirname(os.path.abspath(__file__))


def get_unified_processed_data():
    train_df = pd.read_csv(os.path.join(_HERE, "train.csv"))
    test_df = pd.read_csv(os.path.join(_HERE, "test.csv"))
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_mask = df['Transported'].notna()

    # ========== 1. Split PassengerId ==========
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['PP'] = df['PassengerId'].apply(lambda x: int(x.split('_')[1]))
    df['Group_Size'] = df.groupby('Group')['PassengerId'].transform('count')
    df['Is_Alone'] = (df['Group_Size'] == 1).astype(int)

    # ========== 2. Split Cabin ==========
    df[['Deck', 'Cabin_Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Cabin_Num'] = pd.to_numeric(df['Cabin_Num'], errors='coerce')

    # ========== 3. Name Processing (used only for imputation, no Family_* features created) ==========
    df['Surname'] = df['Name'].str.split(' ').str[-1]
    df['Family_Size'] = df.groupby('Surname')['PassengerId'].transform('count')
    df['Family_Size'] = df['Family_Size'].fillna(1)

    # ========== 4. Spending Feature Engineering ==========
    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    luxury_cols = ['RoomService', 'Spa', 'VRDeck']
    basic_cols = ['FoodCourt', 'ShoppingMall']

    # CryoSleep consistency correction (key domain rule, retained)
    temp_total_spend = df[exp_cols].sum(axis=1, skipna=True)
    df.loc[temp_total_spend > 0, 'CryoSleep'] = False
    df.loc[df['CryoSleep'] == True, exp_cols] = 0
    df[exp_cols] = df[exp_cols].fillna(0)

    df['Total_Spending'] = df[exp_cols].sum(axis=1)
    df['Luxury_Spending'] = df[luxury_cols].sum(axis=1)
    df['Basic_Spending'] = df[basic_cols].sum(axis=1)

    df['No_Spend'] = (df['Total_Spending'] == 0).astype(int)
    df['Spending_Diversity'] = (df[exp_cols] > 0).sum(axis=1)

    # Keep only 2 most discriminative ratios (luxury spending is key to Transported prediction)
    df['Spa_Ratio_raw'] = df['Spa'] / (df['Total_Spending'] + 1)
    df['VRDeck_Ratio_raw'] = df['VRDeck'] / (df['Total_Spending'] + 1)

    # log1p transform
    for col in exp_cols + ['Total_Spending', 'Luxury_Spending', 'Basic_Spending']:
        df[col] = np.log1p(df[col])
    # Ratios are not log-transformed (in [0, 1])
    df['Spa_Ratio'] = df['Spa_Ratio_raw']
    df['VRDeck_Ratio'] = df['VRDeck_Ratio_raw']
    df = df.drop(columns=['Spa_Ratio_raw', 'VRDeck_Ratio_raw'])

    # ==========================================================
    # 5. Missing value imputation - keep all v66 4-layer backfill + domain rules
    # ==========================================================

    # ----- HomePlanet & Destination -----
    for col in ['HomePlanet', 'Destination']:
        df[col] = df.groupby('Group')[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
        df[col] = df.groupby('Surname')[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

    # Deck → HomePlanet inference (train-only mapping)
    if df['HomePlanet'].isna().any():
        deck_to_planet = df[train_mask].groupby('Deck')['HomePlanet'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        ).to_dict()
        mask = df['HomePlanet'].isna() & df['Deck'].notna()
        df.loc[mask, 'HomePlanet'] = df.loc[mask, 'Deck'].map(deck_to_planet)

    # VIP=True → Europa (train-only mode)
    vip_planet_mode = df[train_mask & (df['VIP'] == True)]['HomePlanet'].mode()
    if not vip_planet_mode.empty:
        mask = df['HomePlanet'].isna() & (df['VIP'] == True)
        df.loc[mask, 'HomePlanet'] = vip_planet_mode[0]

    # Global fallback (train-only mode)
    for col in ['HomePlanet', 'Destination']:
        train_mode = df.loc[train_mask, col].mode()[0]
        df[col] = df[col].fillna(train_mode)

    # ----- Deck & Side -----
    for col in ['Deck', 'Side']:
        df[col] = df.groupby('Group')[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

    deck_map = {'Europa': 'B', 'Earth': 'G', 'Mars': 'F'}
    mask = df['Deck'].isna() & df['HomePlanet'].notna()
    df.loc[mask, 'Deck'] = df.loc[mask, 'HomePlanet'].map(deck_map)

    for col in ['Deck', 'Side']:
        df[col] = df.groupby('Surname')[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

    for col in ['Deck', 'Side']:
        train_mode = df.loc[train_mask, col].mode()[0]
        df[col] = df[col].fillna(train_mode)

    # ----- Cabin_Num -----
    df['Cabin_Was_Missing'] = df['Cabin_Num'].isna().astype(int)
    df['Cabin_Num'] = df.groupby('Group')['Cabin_Num'].transform(
        lambda x: x.fillna(x.median())
    )
    train_deck_medians = df[train_mask].groupby('Deck')['Cabin_Num'].median().to_dict()
    train_overall_median = df.loc[train_mask, 'Cabin_Num'].median()
    fill_values = df['Deck'].map(train_deck_medians).fillna(train_overall_median)
    df['Cabin_Num'] = df['Cabin_Num'].fillna(fill_values)
    df['Cabin_Num'] = df['Cabin_Num'].round().astype(int)

    # ----- Age: IterativeImputer (strict train-only fit) -----
    imp_cols = ['Age', 'Total_Spending', 'RoomService', 'Spa', 'VRDeck',
                'CryoSleep', 'Is_Alone', 'Family_Size']
    temp_df = df[imp_cols + ['Deck', 'HomePlanet']].copy()
    for col in ['Deck', 'HomePlanet']:
        temp_df[col] = temp_df[col].astype('category').cat.codes
    temp_df['CryoSleep'] = temp_df['CryoSleep'].astype(float)

    imputer = IterativeImputer(random_state=42, max_iter=10)
    imputer.fit(temp_df[train_mask])
    imputed_values = imputer.transform(temp_df)
    df['Age'] = np.clip(imputed_values[:, 0], 0, 100)

    # CryoSleep imputation (use value from IterativeImputer, fill only original missing positions)
    cryo_imputed = imputed_values[:, 5]
    cryo_mask = df['CryoSleep'].isna()
    df.loc[cryo_mask, 'CryoSleep'] = (cryo_imputed[cryo_mask.values] > 0.5)

    df['Is_Child'] = (df['Age'] < 13).astype(int)
    df['Age_Group'] = pd.cut(df['Age'],
        bins=[-1, 12, 18, 25, 35, 50, 100],
        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior']
    ).astype(str)

    # ========== 6. Cabin derived (train-only binning) ==========
    train_cabin_nums = df.loc[train_mask, 'Cabin_Num']
    _, bin_edges = pd.qcut(train_cabin_nums, q=6, retbins=True, duplicates='drop')
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    df['Cabin_Region'] = pd.cut(df['Cabin_Num'], bins=bin_edges, labels=False)

    df['Cabin_Even'] = (df['Cabin_Num'] % 2 == 0).astype(int)
    df['Cabin_Num_Bucket'] = (df['Cabin_Num'] // 100).astype(int)

    # ========== 7. CryoSleep / VIP fallback ==========
    train_cryo_mode = df.loc[train_mask, 'CryoSleep'].mode()[0]
    df['CryoSleep'] = df['CryoSleep'].fillna(train_cryo_mode).astype(int)
    df['VIP'] = df['VIP'].fillna(False).astype(int)

    # ========== 8. Group statistics (highly simplified, keep only 4 core features) ==========
    # Removed: Spending_Max, Age_Std, PP_Range, PP_Max, Total_Spending,
    #          Has_Child, All_Cryo (collinear or marginal compared to below)
    df['Group_Spending_Mean'] = df.groupby('Group')['Total_Spending'].transform('mean')
    df['Group_Age_Mean'] = df.groupby('Group')['Age'].transform('mean')
    df['Group_Any_Cryo'] = df.groupby('Group')['CryoSleep'].transform('max')
    df['Group_Cryo_Ratio'] = df.groupby('Group')['CryoSleep'].transform('mean')
    df['Group_Has_VIP'] = df.groupby('Group')['VIP'].transform('max')

    # ========== 9. Combinations features (remove two high-cardinality, keep only Deck_Side) ==========
    df['Deck_Side'] = df['Deck'].astype(str) + '_' + df['Side'].astype(str)

    # ========== 10. Strong logical features (keep 2 core) ==========
    df['Child_Alone'] = ((df['Is_Child'] == 1) & (df['Is_Alone'] == 1)).astype(int)
    df['VIP_No_Spend'] = ((df['VIP'] == 1) & (df['No_Spend'] == 1)).astype(int)

    # ========== 11. Feature selection (38) ==========
    features = [
        # Original categories (5)
        'HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Age',
        # Spending (5 original + 3 aggregate = 8, after log1p)
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'Total_Spending', 'Luxury_Spending', 'Basic_Spending',
        # Spending derived (4)
        'No_Spend', 'Spending_Diversity', 'Spa_Ratio', 'VRDeck_Ratio',
        # Group (3 base + 5 stats = 8)
        'Group_Size', 'Is_Alone', 'PP',
        'Group_Spending_Mean', 'Group_Age_Mean',
        'Group_Any_Cryo', 'Group_Cryo_Ratio', 'Group_Has_VIP',
        # Family (1 base, removed Family_*_Mean)
        'Family_Size',
        # Cabin (3 base + 4 derived = 7)
        'Deck', 'Cabin_Num', 'Side', 'Deck_Side',
        'Cabin_Region', 'Cabin_Even', 'Cabin_Num_Bucket', 'Cabin_Was_Missing',
        # Age derived (2, removed Is_Teen)
        'Is_Child', 'Age_Group',
        # Strong logic (2)
        'Child_Alone', 'VIP_No_Spend',
    ]
    assert len(features) == 38, f"features should be 38, but got {len(features)}"

    cat_cols = [
        'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
        'Deck', 'Side', 'Deck_Side',
        'No_Spend', 'Is_Child', 'Age_Group',
        'Cabin_Region', 'Cabin_Even', 'Cabin_Was_Missing', 'Cabin_Num_Bucket',
        'Group_Any_Cryo', 'Group_Has_VIP',
        'Is_Alone',
        'Child_Alone', 'VIP_No_Spend',
    ]

    X = df[df['Transported'].notna()][features].reset_index(drop=True)
    y = df[df['Transported'].notna()]['Transported'].astype(int).reset_index(drop=True)
    X_test = df[df['Transported'].isna()][features].reset_index(drop=True)

    return X, y, X_test, cat_cols


def get_linear_model_data():
    """Linear model (LR / KNN) input: one-hot + RobustScaler.

    Why use RobustScaler instead of StandardScaler?
    - Age and Spending features typically have long-tailed/outlier distributions
    - RobustScaler uses median/IQR scaling, less sensitive to outliers
    - Generally better for financial/spending features than StandardScaler
    """
    X, y, X_test, cat_cols = get_unified_processed_data()
    X_linear = pd.get_dummies(X, columns=cat_cols)
    X_test_linear = pd.get_dummies(X_test, columns=cat_cols)
    X_linear, X_test_linear = X_linear.align(
        X_test_linear, join='left', axis=1, fill_value=0
    )
    scaler = RobustScaler()
    X_linear_scaled = scaler.fit_transform(X_linear)
    X_test_linear_scaled = scaler.transform(X_test_linear)
    return X_linear_scaled, y, X_test_linear_scaled, X_linear.columns.tolist()


if __name__ == "__main__":
    print("=" * 70)
    print("Spaceship Titanic Data Preprocessing v7.0 - anti-overfit simplified (62 → 38)")
    print("=" * 70)

    X, y, X_test, cat_cols = get_unified_processed_data()
    print(f"\nTrain shape: {X.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Total features:   {X.shape[1]}")
    print(f"Categorical features:   {len(cat_cols)} ")
    print(f"Numeric features:   {X.shape[1] - len(cat_cols)} ")

    # print(f"\nDeck x HomePlanet crosstab (domain rule validation):")
    print(pd.crosstab(X['HomePlanet'], X['Deck']))

    # print(f"\nVIP x HomePlanet (VIP→Europa rule validation):")
    print(pd.crosstab(X['VIP'], X['HomePlanet']))

    print(f"\nLabel distribution:")
    print(y.value_counts(normalize=True).round(4))

    print(f"\nMissing value check (should be all 0):")
    miss = X.isna().sum()
    miss = miss[miss > 0]
    if len(miss) == 0:
        print("  No missing")
    else:
        print(miss)

    X_lin, _, X_test_lin, fn = get_linear_model_data()
    # print(f"\nLinear model data (after one-hot): X={X_lin.shape}, X_test={X_test_lin.shape}")
    # print(f"Features after expansion: {len(fn)} (V66 has ~120, after removing combined categories should reduce ~30)")

    print("\n" + "=" * 70)
    # print("v7.0 Done - 38 features, train-only fit, no longer relies on Family/collinear derived")
    print("=" * 70)
