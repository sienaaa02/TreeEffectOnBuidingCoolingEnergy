import os
import json
import joblib
import numpy as np
import pandas as pd
import h5py
import unicodedata
import re

from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold

# ============================================================
# STACKED XGB TEST MODEL:
#   Daily_Trees + Daily_NoTree
#
# Changes relative to previous MAY model:
#   1. Remove Hemisphere feature completely
#   2. Keep City one-hot encoding for day-CV and final model
#   3. Drop City one-hot during city-CV to avoid held-out city dummy issue
#   4. Enforce monotone decrease on BOTH:
#         f_tree_mean  -> -1
#         LAI_TC_mean  -> -1
#
# No-tree rows:
#   f_tree_mean = 0
#   LAI_TC_mean = 0
#
# Purpose:
#   Sensitivity/robustness model to test whether dual monotonicity
#   reduces positive tree-effect artefacts under CMIP6 forcing.
# ============================================================

# =========================
# PATHS
# =========================
FEATURE_CSV = r"E:\UTC_BEM_100cities_loop\UTC_BEM_Emulator_Input_DailySummer_NEW_WEIGHTED.csv"
TARGET_MAT  = r"E:\UTC_BEM_100cities_loop\SummerDaily_ECAC_PYFRIENDLY_Tropical_fixed_WEIGHTED.mat"

OUT_DIR = r"D:\XGBoost_NEW\xgb_stacked_monotone_ftree_LAI"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5

# ============================================================
# TUNING SETTINGS
# ============================================================
N_EST_CANDIDATES = [100, 150, 200, 275, 350, 450, 600, 700]
SELECT_BEST_BY = "day_citymean"

# =========================
# BASE MODEL PARAMS
# =========================
BASE_XGB_PARAMS = dict(
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    eval_metric="rmse"
)

# =========================
# MONOTONICITY SETTINGS
# =========================
USE_MONOTONE_TREE_VARS = True

MONOTONE_CONSTRAINTS = {
    "f_tree_mean": -1,
    "LAI_TC_mean": -1,
}

# =========================
# FEATURE SET
# No RH, no Hemisphere.
# =========================
NUM_VARS = [
    "Lat",
    "Tair_summer_mean_C",
    "DewPoint_summer_mean_C",
    "AirConRoomFraction_mean",
    "f_tree_mean",
    "LAI_TC_mean",
    "GlazingRatio_mean",
    "CDD_proxy",
    "SW_total",
]

# ============================================================
# HELPERS
# ============================================================
def normalize_city_key(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s)


def rmse_np(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)

    if m.sum() == 0:
        return np.nan

    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def mae_np(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)

    if m.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(a[m] - b[m])))


def r2_np(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]

    if len(a) == 0:
        return np.nan

    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)

    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# -------- MAT reader --------
def _read_matlab_string(h5, ref_or_ds):
    if isinstance(ref_or_ds, h5py.Dataset):
        arr = ref_or_ds[()]
    else:
        arr = h5[ref_or_ds][()]

    arr = np.array(arr)

    if arr.dtype == np.uint16:
        return "".join(chr(int(c)) for c in arr.flatten() if int(c) != 0).strip()

    return str(arr)


def load_daily_target_from_ResultsPY(mat_path, source, col=0):
    rows = []

    with h5py.File(mat_path, "r") as f:
        city_cell = f["ResultsPY/City"]
        data_cell = f[f"ResultsPY/{source}"]
        n_city = city_cell.shape[1]

        for i in range(n_city):
            city = _read_matlab_string(f, city_cell[0, i])
            arr = np.squeeze(np.array(f[data_cell[0, i]][()]))

            if arr.shape[0] == 3:
                arr = arr.T

            yv = arr[:, int(col)].astype(float)

            for d in range(len(yv)):
                rows.append((city, d + 1, float(yv[d])))

    return pd.DataFrame(rows, columns=["City", "DayIndex", "y"])


# -------- Design matrix builder --------
def build_X(df, drop_city_dummies=False):
    """
    Build design matrix for the test model.

    day-CV/final:
        drop_city_dummies=False
        features = numeric vars + City one-hot

    city-CV:
        drop_city_dummies=True
        features = numeric vars only
        because held-out cities are unseen during training.
    """
    df = df.copy()

    keep = ["City", "DayIndex"] + NUM_VARS
    missing = [c for c in keep if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    df = df[keep].copy()

    df["City"] = df["City"].astype(str)

    df["DayIndex"] = pd.to_numeric(df["DayIndex"], errors="coerce")
    df = df.dropna(subset=["DayIndex"]).copy()
    df["DayIndex"] = df["DayIndex"].astype(int)

    for c in NUM_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if drop_city_dummies:
        df = df.drop(columns=["City"])
    else:
        df = pd.get_dummies(df, columns=["City"], drop_first=False)

    dayindex = df["DayIndex"].astype(int).to_numpy()
    df = df.drop(columns=["DayIndex"])

    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float64)

    return df, dayindex


def make_monotone_tuple(feature_columns, constraints_dict):
    """
    Create XGBoost monotone constraint tuple.

    Example:
        constraints_dict = {
            "f_tree_mean": -1,
            "LAI_TC_mean": -1
        }
    """
    cons = []

    print("[MONO] Constraints requested:")

    for k, v in constraints_dict.items():
        if k in feature_columns:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: NOT FOUND in feature columns")

    for c in feature_columns:
        cons.append(int(constraints_dict.get(c, 0)))

    n_constrained = sum(1 for x in cons if x != 0)
    print(f"[MONO] Total constrained columns: {n_constrained}")

    return tuple(cons)


# -------- 5-fold day assignment within each city-state --------
def assign_day_folds_within_city_state(df, n_folds=5, seed=42):
    """
    Assign folds by city and day, with trees/no-tree rows for the
    same city-day kept in the same fold.
    """
    rng = np.random.default_rng(seed)
    fold_id = np.full(len(df), -1, dtype=int)

    city_day_fold_map = {}

    for city_key in df["City_key"].unique():
        days = np.array(sorted(df[df["City_key"] == city_key]["DayIndex"].unique()))

        rng.shuffle(days)
        day_folds = np.array_split(days, n_folds)

        day_to_fold = {}

        for k, day_subset in enumerate(day_folds):
            for d in day_subset.tolist():
                day_to_fold[d] = k

        city_day_fold_map[city_key] = day_to_fold

    for idx, row in df.iterrows():
        ck = row["City_key"]
        d = int(row["DayIndex"])
        fold_id[df.index.get_loc(idx)] = city_day_fold_map[ck][d]

    if np.any(fold_id < 0):
        raise RuntimeError("Some rows were not assigned a day fold.")

    return fold_id


# -------- metric calculators --------
def compute_daily_metrics(y_true, y_pred):
    return {
        "rmse": rmse_np(y_true, y_pred),
        "mae": mae_np(y_true, y_pred),
        "r2": r2_np(y_true, y_pred),
    }


def compute_citymean_table(df_meta, y_true, y_pred):
    tmp = pd.DataFrame({
        "City": df_meta["City"].astype(str).values,
        "City_key": df_meta["City_key"].astype(str).values,
        "state_label": df_meta["state_label"].astype(str).values,
        "DayIndex": df_meta["DayIndex"].astype(int).values,
        "ECAC_obs_daily": np.asarray(y_true, float),
        "ECAC_pred_daily": np.asarray(y_pred, float),
    })

    citymean = (
        tmp.groupby(["City_key", "state_label"], as_index=False)
           .agg(
               City=("City", "first"),
               ECAC_obs_summer_mean=("ECAC_obs_daily", "mean"),
               ECAC_pred_summer_mean=("ECAC_pred_daily", "mean"),
               n_days=("ECAC_obs_daily", "size"),
           )
    )

    return citymean


def compute_citymean_metrics(df_citymean):
    return {
        "rmse": rmse_np(
            df_citymean["ECAC_obs_summer_mean"],
            df_citymean["ECAC_pred_summer_mean"],
        ),
        "mae": mae_np(
            df_citymean["ECAC_obs_summer_mean"],
            df_citymean["ECAC_pred_summer_mean"],
        ),
        "r2": r2_np(
            df_citymean["ECAC_obs_summer_mean"],
            df_citymean["ECAC_pred_summer_mean"],
        ),
    }


# -------- exports --------
def export_predictions_and_metrics(tag, df_meta, y_true, y_pred, out_dir):
    y_pred = np.clip(np.asarray(y_pred, float), 0.0, None)

    df_daily = pd.DataFrame({
        "City": df_meta["City"].astype(str).values,
        "City_key": df_meta["City_key"].astype(str).values,
        "state_label": df_meta["state_label"].astype(str).values,
        "DayIndex": df_meta["DayIndex"].astype(int).values,
        "ECAC_obs_daily": np.asarray(y_true, float),
        "ECAC_pred_daily": y_pred,
    })

    out_daily = os.path.join(out_dir, f"{tag}_QA_OBS_vs_PRED_DAILY.csv")
    df_daily.to_csv(out_daily, index=False)

    df_citymean = compute_citymean_table(df_meta, y_true, y_pred)
    out_citymean = os.path.join(out_dir, f"{tag}_QA_OBS_vs_PRED_CITYMEAN.csv")
    df_citymean.to_csv(out_citymean, index=False)

    daily_metrics = compute_daily_metrics(
        df_daily["ECAC_obs_daily"],
        df_daily["ECAC_pred_daily"],
    )

    citymean_metrics = compute_citymean_metrics(df_citymean)

    metrics = {
        "daily": {
            "rmse": daily_metrics["rmse"],
            "mae": daily_metrics["mae"],
            "r2": daily_metrics["r2"],
            "n_rows": int(len(df_daily)),
            "n_city_states": int(df_daily.groupby(["City_key", "state_label"]).ngroups),
        },
        "citymean": {
            "rmse": citymean_metrics["rmse"],
            "mae": citymean_metrics["mae"],
            "r2": citymean_metrics["r2"],
            "n_city_states": int(len(df_citymean)),
        },
    }

    out_metrics = os.path.join(out_dir, f"{tag}_METRICS.json")

    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[QA] {tag} | "
        f"DAILY RMSE={metrics['daily']['rmse']:.4g}, "
        f"MAE={metrics['daily']['mae']:.4g}, "
        f"R2={metrics['daily']['r2']:.4g} | "
        f"CITYMEAN RMSE={metrics['citymean']['rmse']:.4g}, "
        f"MAE={metrics['citymean']['mae']:.4g}, "
        f"R2={metrics['citymean']['r2']:.4g}"
    )

    return metrics


# -------- core fold fit/predict --------
def fit_predict_fold(df_train, df_test, n_estimators, drop_city_dummies=False):
    Xtr, _ = build_X(df_train, drop_city_dummies=drop_city_dummies)
    ytr = df_train["y"].to_numpy(float)

    Xte, _ = build_X(df_test, drop_city_dummies=drop_city_dummies)
    yte = df_test["y"].to_numpy(float)

    mask_tr = np.isfinite(ytr) & np.isfinite(Xtr.to_numpy()).all(axis=1)
    Xtr = Xtr.loc[mask_tr].reset_index(drop=True)
    ytr = ytr[mask_tr]
    df_train_aligned = df_train.loc[mask_tr].reset_index(drop=True)

    mask_te = np.isfinite(yte) & np.isfinite(Xte.to_numpy()).all(axis=1)
    Xte = Xte.loc[mask_te].reset_index(drop=True)
    yte = yte[mask_te]
    df_test_aligned = df_test.loc[mask_te].reset_index(drop=True)

    feature_cols = list(Xtr.columns)
    Xte = Xte.reindex(columns=feature_cols, fill_value=0.0)

    mono = None

    if USE_MONOTONE_TREE_VARS:
        mono = make_monotone_tuple(feature_cols, MONOTONE_CONSTRAINTS)

    params = BASE_XGB_PARAMS.copy()
    params["n_estimators"] = n_estimators

    if mono is not None:
        model = XGBRegressor(**params, monotone_constraints=mono)
    else:
        model = XGBRegressor(**params)

    model.fit(Xtr, ytr, verbose=False)

    pred = np.clip(model.predict(Xte), 0.0, None)

    daily_metrics = compute_daily_metrics(yte, pred)
    df_citymean = compute_citymean_table(df_test_aligned, yte, pred)
    citymean_metrics = compute_citymean_metrics(df_citymean)

    metrics = {
        "daily_rmse": daily_metrics["rmse"],
        "daily_mae": daily_metrics["mae"],
        "daily_r2": daily_metrics["r2"],
        "citymean_rmse": citymean_metrics["rmse"],
        "citymean_mae": citymean_metrics["mae"],
        "citymean_r2": citymean_metrics["r2"],
        "n_rows": int(len(yte)),
        "n_city_states": int(df_test_aligned.groupby(["City_key", "state_label"]).ngroups),
    }

    return pred, yte, df_test_aligned, df_citymean, metrics, feature_cols, model


# -------- summary helper --------
def summarize_cv_table(df_cv, prefix, out_dir):
    summary = (
        df_cv.groupby("n_estimators", as_index=False)
             .agg(
                 daily_rmse_mean=("daily_rmse", "mean"),
                 daily_rmse_std=("daily_rmse", "std"),
                 daily_mae_mean=("daily_mae", "mean"),
                 daily_mae_std=("daily_mae", "std"),
                 daily_r2_mean=("daily_r2", "mean"),
                 daily_r2_std=("daily_r2", "std"),
                 citymean_rmse_mean=("citymean_rmse", "mean"),
                 citymean_rmse_std=("citymean_rmse", "std"),
                 citymean_mae_mean=("citymean_mae", "mean"),
                 citymean_mae_std=("citymean_mae", "std"),
                 citymean_r2_mean=("citymean_r2", "mean"),
                 citymean_r2_std=("citymean_r2", "std"),
             )
             .sort_values("citymean_rmse_mean", ascending=True)
             .reset_index(drop=True)
    )

    out_csv = os.path.join(out_dir, f"{prefix}_summary_by_nestimators.csv")
    summary.to_csv(out_csv, index=False)

    return summary


def choose_best_n(day_summary, city_summary, rule="day_citymean"):
    if rule == "day_citymean":
        tmp = day_summary.sort_values(
            ["citymean_rmse_mean", "daily_rmse_mean", "n_estimators"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        return int(tmp.loc[0, "n_estimators"])

    elif rule == "day_daily":
        tmp = day_summary.sort_values(
            ["daily_rmse_mean", "citymean_rmse_mean", "n_estimators"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        return int(tmp.loc[0, "n_estimators"])

    elif rule == "city_citymean":
        tmp = city_summary.sort_values(
            ["citymean_rmse_mean", "daily_rmse_mean", "n_estimators"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        return int(tmp.loc[0, "n_estimators"])

    elif rule == "city_daily":
        tmp = city_summary.sort_values(
            ["daily_rmse_mean", "citymean_rmse_mean", "n_estimators"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        return int(tmp.loc[0, "n_estimators"])

    else:
        raise ValueError("Unsupported SELECT_BEST_BY rule.")


# ============================================================
# (0) LOAD + BUILD STACKED TABLE
# ============================================================
print("\n[0/7] Load FEATURE_CSV + MAT targets and build stacked table...")

df_feat = pd.read_csv(FEATURE_CSV, sep=None, engine="python")
df_feat.columns = [str(c).strip() for c in df_feat.columns]

df_feat["City"] = df_feat["City"].astype(str)

df_feat["DayIndex"] = pd.to_numeric(df_feat["DayIndex"], errors="coerce")
df_feat = df_feat.dropna(subset=["City", "DayIndex"]).copy()
df_feat["DayIndex"] = df_feat["DayIndex"].astype(int)

df_feat["City_key"] = df_feat["City"].apply(normalize_city_key)

# Check required feature columns
missing_feat = [c for c in NUM_VARS if c not in df_feat.columns]

if missing_feat:
    raise ValueError(f"FEATURE_CSV missing NUM_VARS: {missing_feat}")

for c in NUM_VARS:
    df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

df_trees = load_daily_target_from_ResultsPY(
    TARGET_MAT,
    source="Daily_Trees",
    col=0,
)

df_notree = load_daily_target_from_ResultsPY(
    TARGET_MAT,
    source="Daily_NoTree",
    col=0,
)

# ---- baseline trees block ----
df_block_trees = df_feat.merge(df_trees, on=["City", "DayIndex"], how="inner")
df_block_trees = df_block_trees.dropna(subset=["y"]).reset_index(drop=True)
df_block_trees["state_label"] = "trees"

n_feat = len(df_feat)
n_merged = len(df_block_trees)
pct_kept = 100.0 * n_merged / n_feat

print(f"[CHECK] Trees merge: {n_feat} feat rows -> {n_merged} merged rows ({pct_kept:.1f}% kept)")

assert pct_kept >= 95.0, (
    f"Trees merge kept only {pct_kept:.1f}% of feature rows. "
    f"Expected >=95%. Check City name alignment and DayIndex coverage."
)

# ---- no-tree block ----
df_block_notree = df_feat.copy()
df_block_notree["f_tree_mean"] = 0.0
df_block_notree["LAI_TC_mean"] = 0.0

df_block_notree = df_block_notree.merge(df_notree, on=["City", "DayIndex"], how="inner")
df_block_notree = df_block_notree.dropna(subset=["y"]).reset_index(drop=True)
df_block_notree["state_label"] = "notree"

n_merged_nt = len(df_block_notree)
pct_kept_nt = 100.0 * n_merged_nt / n_feat

print(f"[CHECK] NoTree merge: {n_feat} feat rows -> {n_merged_nt} merged rows ({pct_kept_nt:.1f}% kept)")

assert pct_kept_nt >= 95.0, (
    f"NoTree merge kept only {pct_kept_nt:.1f}% of feature rows. "
    f"Expected >=95%. Check City name alignment and DayIndex coverage."
)

# Ensure both blocks cover same city-day pairs
pairs_trees = set(zip(df_block_trees["City"], df_block_trees["DayIndex"]))
pairs_notree = set(zip(df_block_notree["City"], df_block_notree["DayIndex"]))

only_in_trees = pairs_trees - pairs_notree
only_in_notree = pairs_notree - pairs_trees

if only_in_trees or only_in_notree:
    raise AssertionError(
        f"Trees and NoTree blocks cover different city+day pairs.\n"
        f"  Only in trees:  {len(only_in_trees)} pairs\n"
        f"  Only in notree: {len(only_in_notree)} pairs\n"
        f"Sample only-in-trees:  {list(only_in_trees)[:5]}\n"
        f"Sample only-in-notree: {list(only_in_notree)[:5]}"
    )

print("[CHECK] Trees and NoTree blocks cover identical city+day pairs. OK.")

# ---- stack ----
df_all = pd.concat([df_block_trees, df_block_notree], axis=0, ignore_index=True)

print("[OK] Stacked rows:", len(df_all))
print("[OK] Cities:", df_all["City_key"].nunique())
print("[OK] State counts:")
print(df_all["state_label"].value_counts())


# ============================================================
# (1) BUILD 5-FOLD DAY SPLITS
# ============================================================
print("\n[1/7] Build 5-fold DAY-CV split...")

df_all = df_all.copy()
df_all["day_fold"] = assign_day_folds_within_city_state(
    df_all,
    n_folds=N_FOLDS,
    seed=RANDOM_SEED,
)

print(df_all["day_fold"].value_counts().sort_index())


# ============================================================
# (2) BUILD 5-FOLD CITY SPLITS
# ============================================================
print("\n[2/7] Build 5-fold CITY-CV split...")

all_city_keys = df_all["City_key"].astype(str).values
gkf = GroupKFold(n_splits=N_FOLDS)
city_splits = list(gkf.split(df_all, groups=all_city_keys))

print(f"[OK] Built {len(city_splits)} city folds.")


# ============================================================
# (3) 5-FOLD DAY-CV TUNING
# ============================================================
print("\n[3/7] 5-fold DAY-CV tuning...")

rows_day = []
best_day_daily_predictions = {}
best_day_citymean_predictions = {}

for n_est in N_EST_CANDIDATES:
    print(f"\n[DAY-CV] Testing n_estimators = {n_est}")

    fold_daily_preds = []
    fold_citymean_preds = []

    for fold in range(N_FOLDS):
        df_train = df_all[df_all["day_fold"] != fold].reset_index(drop=True)
        df_test = df_all[df_all["day_fold"] == fold].reset_index(drop=True)

        pred, yte, df_meta, df_citymean, metrics, _, _ = fit_predict_fold(
            df_train,
            df_test,
            n_estimators=n_est,
            drop_city_dummies=False,
        )

        row = {
            "n_estimators": n_est,
            "fold": fold,
            "daily_rmse": metrics["daily_rmse"],
            "daily_mae": metrics["daily_mae"],
            "daily_r2": metrics["daily_r2"],
            "citymean_rmse": metrics["citymean_rmse"],
            "citymean_mae": metrics["citymean_mae"],
            "citymean_r2": metrics["citymean_r2"],
            "n_rows": metrics["n_rows"],
            "n_city_states": metrics["n_city_states"],
        }

        rows_day.append(row)

        fold_daily_df = pd.DataFrame({
            "City": df_meta["City"].values,
            "City_key": df_meta["City_key"].values,
            "state_label": df_meta["state_label"].values,
            "DayIndex": df_meta["DayIndex"].values,
            "ECAC_obs_daily": yte,
            "ECAC_pred_daily": pred,
            "fold": fold,
            "n_estimators": n_est,
        })

        fold_daily_preds.append(fold_daily_df)

        df_citymean_export = df_citymean.copy()
        df_citymean_export["fold"] = fold
        df_citymean_export["n_estimators"] = n_est

        fold_citymean_preds.append(df_citymean_export)

        print(
            f"  fold={fold} | "
            f"DAILY RMSE={metrics['daily_rmse']:.4g}, "
            f"MAE={metrics['daily_mae']:.4g}, "
            f"R2={metrics['daily_r2']:.4g} | "
            f"CITYMEAN RMSE={metrics['citymean_rmse']:.4g}, "
            f"MAE={metrics['citymean_mae']:.4g}, "
            f"R2={metrics['citymean_r2']:.4g}"
        )

    best_day_daily_predictions[n_est] = pd.concat(fold_daily_preds, axis=0, ignore_index=True)
    best_day_citymean_predictions[n_est] = pd.concat(fold_citymean_preds, axis=0, ignore_index=True)

df_day_cv = pd.DataFrame(rows_day)

out_day_cv = os.path.join(OUT_DIR, "DAYCV_fold_metrics_by_nestimators.csv")
df_day_cv.to_csv(out_day_cv, index=False)

day_summary = summarize_cv_table(df_day_cv, prefix="DAYCV", out_dir=OUT_DIR)

print("\n[DAY-CV SUMMARY]")
print(day_summary)

best_n_day_daily = int(day_summary.sort_values("daily_rmse_mean").iloc[0]["n_estimators"])
best_n_day_citymean = int(day_summary.sort_values("citymean_rmse_mean").iloc[0]["n_estimators"])

print(f"[OK] Best n_estimators by DAY-CV DAILY RMSE:   {best_n_day_daily}")
print(f"[OK] Best n_estimators by DAY-CV CITYMEAN RMSE: {best_n_day_citymean}")


# ============================================================
# (4) 5-FOLD CITY-CV TUNING
# ============================================================
print("\n[4/7] 5-fold CITY-CV tuning...")

rows_city = []
best_city_daily_predictions = {}
best_city_citymean_predictions = {}

for n_est in N_EST_CANDIDATES:
    print(f"\n[CITY-CV] Testing n_estimators = {n_est}")

    fold_daily_preds = []
    fold_citymean_preds = []

    for fold, (tr_idx, te_idx) in enumerate(city_splits):
        df_train = df_all.iloc[tr_idx].reset_index(drop=True)
        df_test = df_all.iloc[te_idx].reset_index(drop=True)

        pred, yte, df_meta, df_citymean, metrics, _, _ = fit_predict_fold(
            df_train,
            df_test,
            n_estimators=n_est,
            drop_city_dummies=True,
        )

        row = {
            "n_estimators": n_est,
            "fold": fold,
            "daily_rmse": metrics["daily_rmse"],
            "daily_mae": metrics["daily_mae"],
            "daily_r2": metrics["daily_r2"],
            "citymean_rmse": metrics["citymean_rmse"],
            "citymean_mae": metrics["citymean_mae"],
            "citymean_r2": metrics["citymean_r2"],
            "n_rows": metrics["n_rows"],
            "n_city_states": metrics["n_city_states"],
        }

        rows_city.append(row)

        fold_daily_df = pd.DataFrame({
            "City": df_meta["City"].values,
            "City_key": df_meta["City_key"].values,
            "state_label": df_meta["state_label"].values,
            "DayIndex": df_meta["DayIndex"].values,
            "ECAC_obs_daily": yte,
            "ECAC_pred_daily": pred,
            "fold": fold,
            "n_estimators": n_est,
        })

        fold_daily_preds.append(fold_daily_df)

        df_citymean_export = df_citymean.copy()
        df_citymean_export["fold"] = fold
        df_citymean_export["n_estimators"] = n_est

        fold_citymean_preds.append(df_citymean_export)

        print(
            f"  fold={fold} | "
            f"DAILY RMSE={metrics['daily_rmse']:.4g}, "
            f"MAE={metrics['daily_mae']:.4g}, "
            f"R2={metrics['daily_r2']:.4g} | "
            f"CITYMEAN RMSE={metrics['citymean_rmse']:.4g}, "
            f"MAE={metrics['citymean_mae']:.4g}, "
            f"R2={metrics['citymean_r2']:.4g}"
        )

    best_city_daily_predictions[n_est] = pd.concat(fold_daily_preds, axis=0, ignore_index=True)
    best_city_citymean_predictions[n_est] = pd.concat(fold_citymean_preds, axis=0, ignore_index=True)

df_city_cv = pd.DataFrame(rows_city)

out_city_cv = os.path.join(OUT_DIR, "CITYCV_fold_metrics_by_nestimators.csv")
df_city_cv.to_csv(out_city_cv, index=False)

city_summary = summarize_cv_table(df_city_cv, prefix="CITYCV", out_dir=OUT_DIR)

print("\n[CITY-CV SUMMARY]")
print(city_summary)

best_n_city_daily = int(city_summary.sort_values("daily_rmse_mean").iloc[0]["n_estimators"])
best_n_city_citymean = int(city_summary.sort_values("citymean_rmse_mean").iloc[0]["n_estimators"])

print(f"[OK] Best n_estimators by CITY-CV DAILY RMSE:   {best_n_city_daily}")
print(f"[OK] Best n_estimators by CITY-CV CITYMEAN RMSE: {best_n_city_citymean}")


# ============================================================
# (5) SELECT FINAL n_estimators
# ============================================================
print("\n[5/7] Select final n_estimators...")

BEST_N_EST = choose_best_n(day_summary, city_summary, rule=SELECT_BEST_BY)

print(f"[OK] Final selected n_estimators = {BEST_N_EST} (criterion = {SELECT_BEST_BY})")

best_day_daily_predictions[BEST_N_EST].to_csv(
    os.path.join(OUT_DIR, f"BEST_DAYCV_DAILY_predictions_STACKED_n{BEST_N_EST}.csv"),
    index=False,
)

best_day_citymean_predictions[BEST_N_EST].to_csv(
    os.path.join(OUT_DIR, f"BEST_DAYCV_CITYMEAN_predictions_STACKED_n{BEST_N_EST}.csv"),
    index=False,
)

best_city_daily_predictions[BEST_N_EST].to_csv(
    os.path.join(OUT_DIR, f"BEST_CITYCV_DAILY_predictions_STACKED_n{BEST_N_EST}.csv"),
    index=False,
)

best_city_citymean_predictions[BEST_N_EST].to_csv(
    os.path.join(OUT_DIR, f"BEST_CITYCV_CITYMEAN_predictions_STACKED_n{BEST_N_EST}.csv"),
    index=False,
)

selection_summary = {
    "N_EST_CANDIDATES": N_EST_CANDIDATES,
    "SELECT_BEST_BY": SELECT_BEST_BY,
    "BEST_N_EST": int(BEST_N_EST),
    "STACKING": "Daily_Trees + Daily_NoTree",
    "CATEGORICAL_FEATURES_DAYCV_FINAL": ["City"],
    "CATEGORICAL_FEATURES_CITYCV": [],
    "HEMISPHERE_INCLUDED": False,
    "USE_MONOTONE_TREE_VARS": USE_MONOTONE_TREE_VARS,
    "MONOTONE_CONSTRAINTS": MONOTONE_CONSTRAINTS,
    "NOTREE_FEATURE_OVERRIDE": {
        "f_tree_mean": 0.0,
        "LAI_TC_mean": 0.0,
    },
}

with open(os.path.join(OUT_DIR, "selected_n_estimators_summary_STACKED.json"), "w") as f:
    json.dump(selection_summary, f, indent=2)


# ============================================================
# (6) TRAIN FINAL MODEL ON 100% STACKED DATA
# ============================================================
print("\n[6/7] Train FINAL model on 100% stacked rows...")

X_full, _ = build_X(df_all, drop_city_dummies=False)
y_full = df_all["y"].to_numpy(float)

mask_full = np.isfinite(y_full) & np.isfinite(X_full.to_numpy()).all(axis=1)

X_full = X_full.loc[mask_full].reset_index(drop=True)
y_full = y_full[mask_full]
df_full_aligned = df_all.loc[mask_full].reset_index(drop=True)

final_feature_cols = list(X_full.columns)

monoF = None

if USE_MONOTONE_TREE_VARS:
    monoF = make_monotone_tuple(final_feature_cols, MONOTONE_CONSTRAINTS)

final_params = BASE_XGB_PARAMS.copy()
final_params["n_estimators"] = BEST_N_EST

if monoF is not None:
    m_final = XGBRegressor(**final_params, monotone_constraints=monoF)
else:
    m_final = XGBRegressor(**final_params)

m_final.fit(X_full, y_full, verbose=False)

tag_final = f"xgb_STACKED_TreesPlusNoTree_CITYFE_dualmonotone_noHemisphere_FINAL_ALLDATA_n{BEST_N_EST}"

out_model = os.path.join(OUT_DIR, f"{tag_final}.joblib")
out_cols = os.path.join(OUT_DIR, f"{tag_final}_feature_columns.json")

joblib.dump(m_final, out_model)

with open(out_cols, "w") as f:
    json.dump(final_feature_cols, f, indent=2)

print("[OK] Saved FINAL model:", out_model)
print("[OK] Saved FINAL cols :", out_cols)

pred_full = np.clip(m_final.predict(X_full), 0.0, None)

export_predictions_and_metrics(
    tag=f"{tag_final}_TRAINFIT",
    df_meta=df_full_aligned[["City", "City_key", "state_label", "DayIndex"]],
    y_true=y_full,
    y_pred=pred_full,
    out_dir=OUT_DIR,
)


# ============================================================
# (7) CANONICAL EXPORT ON STACKED FEATURE UNIVERSE
# ============================================================
print("\n[7/7] Canonical export on stacked feature universe...")

df_base_trees = df_feat.copy()
df_base_trees["state_label"] = "trees"

df_base_notree = df_feat.copy()
df_base_notree["f_tree_mean"] = 0.0
df_base_notree["LAI_TC_mean"] = 0.0
df_base_notree["state_label"] = "notree"

df_base_stacked = pd.concat([df_base_trees, df_base_notree], axis=0, ignore_index=True)

Xb, _ = build_X(df_base_stacked, drop_city_dummies=False)
Xb = Xb.reindex(columns=final_feature_cols, fill_value=0.0)
Xb = Xb.astype(np.float64)

pred_b = np.clip(m_final.predict(Xb), 0.0, None)

df_canon = pd.DataFrame({
    "City": df_base_stacked["City"].astype(str),
    "City_key": df_base_stacked["City_key"].astype(str),
    "state_label": df_base_stacked["state_label"].astype(str),
    "DayIndex": df_base_stacked["DayIndex"].astype(int),
    "ECAC_pred_daily": pred_b,
})

out_canon_daily = os.path.join(OUT_DIR, f"{tag_final}_CANON_STACKED_daily.csv")
df_canon.to_csv(out_canon_daily, index=False)

print("[OK] Saved canonical stacked daily:", out_canon_daily)

df_canon_citymean = (
    df_canon.groupby(["City_key", "state_label"], as_index=False)
            .agg(
                City=("City", "first"),
                ECAC_pred_summer_mean=("ECAC_pred_daily", "mean"),
                n_days=("ECAC_pred_daily", "size"),
            )
)

out_canon_citymean = os.path.join(OUT_DIR, f"{tag_final}_CANON_STACKED_citymean.csv")
df_canon_citymean.to_csv(out_canon_citymean, index=False)

print("[OK] Saved canonical stacked citymean:", out_canon_citymean)

print("\nDONE.")
print("Outputs are in:", OUT_DIR)