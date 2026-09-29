"""
Extrapolation_Validation_RangeHoldout.py

Companion to XGB_model_training (STACKED, AllTreeLevels). Answers a
specific question raised before trusting an f_tree sweep out to +30%:
"how well does this model extrapolate PAST the f_tree range it was
actually trained on?" -- as opposed to the random day/city CV already
reported, which only validates INTERPOLATION within the observed joint
feature distribution.

Two complementary experiments, both using the exact same feature set,
monotone constraints, and base XGB params as the production model (so
results are directly comparable to it):

  EXPERIMENT A -- "range holdout" (the main event, most relevant to a
  +30% sweep). We progressively restrict the TRAINING f_tree range and
  check how well the model recovers ground truth BEYOND that ceiling,
  using ONLY the pattern learned up to the ceiling:
    A1: train on {NoTree, Trees} only        (ceiling ~= each city's own
        baseline f_tree) -> evaluate on the REAL, held-out Plus10 AND
        Plus20 rows (extrapolation distance ~5-20 percentage points,
        varies by city because of the achieved-vs-target capping).
    A2: train on {NoTree, Trees, Plus10}     (ceiling ~= each city's own
        achieved Plus10 f_tree) -> evaluate on the REAL, held-out Plus20
        rows (a further, smaller extrapolation step).
  For every held-out row we compute extrap_dist = that row's achieved
  f_tree_mean minus THAT CITY's own maximum f_tree_mean actually seen in
  the restricted training set -- i.e., the real per-city extrapolation
  distance, not a nominal percentage. Pooling A1+A2 gives an empirical
  "prediction error vs. extrapolation distance" curve. That's the single
  most useful number for judging a further step out to +30%: if error
  grows roughly linearly (or stays flat) with distance up to ~20pp, a
  linear/flat projection to ~30pp is a defensible (if still approximate)
  bound. If error blows up non-linearly near the observed ceiling, that's
  a strong warning sign against sweeping to +30% at all without a
  different (e.g. parametric/physical) extrapolation model for the
  f_tree axis specifically.

  EXPERIMENT B -- "leave-one-out on genuinely-uncapped cities" (secondary,
  cheaper, answers a narrower question: does cross-city sharing work at
  all, for cities whose OWN high-f_tree behavior is hidden but the
  training range itself still includes other cities' Plus20 data). This
  is the leave-out test described earlier in the project chat. It does
  NOT tell you anything about extrapolating past +20% -- only Experiment
  A does that -- but it's a useful sanity check on the City-dummy +
  monotone-GBM generalization mechanism in general.

Also produces the "cheap sanity check" diagnostic: for a handful of
representative cities (a mix of capped/uncapped, NH/SH/TROP), sweep the
REDUCED (range-restricted) model's prediction across f_tree_mean out to
+30 percentage points above that city's Trees baseline, holding every
other feature fixed at that city's own Trees-row values, and overlay the
REAL held-out Plus10/Plus20 points. A model that's extrapolating well
should pass close to those real points; a model that's just plateauing
(the classic GBM extrapolation failure mode) will visibly flatten out
before reaching them.

Nothing here retrains or replaces the production model -- this is a
diagnostic-only companion script. Run it once, read the printed summary
and look at the plots, then decide whether a +30% sweep needs a
different extrapolation strategy for the f_tree/LAI axis before it goes
into the paper.
"""

import os
import json
import numpy as np
import pandas as pd
import unicodedata
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold

# =========================
# PATHS -- match XGB_model_training_AllTreeLevels.py
# =========================
STACKED_CSV = r"E:\Tree_paper_Code\STACKED_AllTreeLevels_n36668.csv"
SELECTED_N_EST_JSON = r"E:\Tree_paper_code\XGBoost_NEW\xgb_stacked_monotone_ftree_LAI_AllTreeLevels\selected_n_estimators_summary_STACKED.json"

OUT_DIR = r"E:\Tree_paper_code\XGBoost_NEW\ExtrapolationValidation_AllTreeLevels"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS_EXPERIMENT_B = 5

# Fallback n_estimators if the production model's selection JSON isn't
# found (keeps this script runnable standalone).
FALLBACK_N_EST = 350

# "Reached target" threshold for Experiment B's uncapped-city subset:
# achieved Plus20 f_tree_mean >= this fraction of (Trees f_tree_mean +
# 0.20), matching the "safe_ceiling_95pct" convention already used for
# the achieved-vs-target QA in this project.
REACHED_TARGET_FRAC = 0.95

# How far past each city's Trees baseline to sweep in the per-city
# diagnostic curves (percentage points of f_tree_mean, additive).
SWEEP_MAX_DELTA = 0.30

# =========================
# FEATURE SET -- must match the production model exactly
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
    eval_metric="rmse",
)

USE_MONOTONE_TREE_VARS = True
MONOTONE_CONSTRAINTS = {
    "f_tree_mean": -1,
    "LAI_TC_mean": -1,
}

EXPECTED_STATES = {"trees", "notree", "plus10", "plus20"}

# ============================================================
# HELPERS -- copied verbatim from XGB_model_training_AllTreeLevels.py
# so results are directly comparable to the production model.
# ============================================================
def normalize_city_key(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s)


def rmse_np(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan


def mae_np(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() else np.nan


def bias_np(a_pred, b_obs):
    # mean signed error, pred - obs. Positive = model OVER-predicts ECAC
    # (i.e., under-predicts the cooling benefit) at that point.
    a = np.asarray(a_pred, float); b = np.asarray(b_obs, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(a[m] - b[m])) if m.sum() else np.nan


def r2_np(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) == 0:
        return np.nan
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((b - np.mean(b)) ** 2)   # fixed: variance of b (observed), not a
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def build_X(df, feature_cols_ref=None):
    """Design matrix: NUM_VARS + City one-hot (matches the production
    model's DAY-CV/final config, drop_city_dummies=False), since that's
    the model whose extrapolation behavior we're actually validating."""
    df = df.copy()
    keep = ["City"] + NUM_VARS
    df = df[keep].copy()
    df["City"] = df["City"].astype(str)
    for c in NUM_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = pd.get_dummies(df, columns=["City"], drop_first=False)
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    if feature_cols_ref is not None:
        df = df.reindex(columns=feature_cols_ref, fill_value=0.0)
    return df


def make_monotone_tuple(feature_columns, constraints_dict):
    return tuple(int(constraints_dict.get(c, 0)) for c in feature_columns)


def fit_model(df_train, n_estimators):
    Xtr = build_X(df_train)
    ytr = df_train["y"].to_numpy(float)
    mask = np.isfinite(ytr) & np.isfinite(Xtr.to_numpy()).all(axis=1)
    Xtr = Xtr.loc[mask].reset_index(drop=True)
    ytr = ytr[mask]
    feature_cols = list(Xtr.columns)

    mono = make_monotone_tuple(feature_cols, MONOTONE_CONSTRAINTS) if USE_MONOTONE_TREE_VARS else None
    params = BASE_XGB_PARAMS.copy()
    params["n_estimators"] = n_estimators
    model = XGBRegressor(**params, monotone_constraints=mono) if mono is not None else XGBRegressor(**params)
    model.fit(Xtr, ytr, verbose=False)
    return model, feature_cols


def predict_with(model, feature_cols, df_eval):
    Xe = build_X(df_eval, feature_cols_ref=feature_cols)
    pred = np.clip(model.predict(Xe), 0.0, None)
    return pred


def get_n_estimators():
    try:
        with open(SELECTED_N_EST_JSON, "r") as f:
            j = json.load(f)
        n = int(j["BEST_N_EST"])
        print(f"[OK] Using production model's selected n_estimators = {n}")
        return n
    except Exception as e:
        print(f"[WARN] Could not read {SELECTED_N_EST_JSON} ({e}); "
              f"falling back to n_estimators = {FALLBACK_N_EST}")
        return FALLBACK_N_EST


# ============================================================
# (0) LOAD DATA -- same loading/validation logic as the production script
# ============================================================
print("\n[0] Loading STACKED_CSV ...")
df_all = pd.read_csv(STACKED_CSV, sep=None, engine="python")
df_all.columns = [str(c).strip() for c in df_all.columns]
df_all["City"] = df_all["City"].astype(str)
df_all["DayIndex"] = pd.to_numeric(df_all["DayIndex"], errors="coerce")
df_all = df_all.dropna(subset=["City", "DayIndex", "y"]).copy()
df_all["DayIndex"] = df_all["DayIndex"].astype(int)

if "City_key" not in df_all.columns:
    df_all["City_key"] = df_all["City"].apply(normalize_city_key)
if "state_label" not in df_all.columns:
    df_all["state_label"] = df_all["TreeState"].astype(str).str.strip().str.lower()

for c in NUM_VARS:
    df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

found_states = set(df_all["state_label"].unique())
print("[OK] Rows:", len(df_all), "| Cities:", df_all["City_key"].nunique(), "| States:", found_states)
if found_states != EXPECTED_STATES:
    print(f"[WARN] state_label values {found_states} != expected {EXPECTED_STATES}. Continuing anyway.")

N_EST = get_n_estimators()

# Per-city Trees-baseline f_tree_mean, used as the sweep origin and as
# the reference point for "achieved delta" everywhere below.
trees_ftree0 = (
    df_all.loc[df_all.state_label == "trees"]
          .groupby("City_key")["f_tree_mean"].mean()
)
df_all["ftree_delta_vs_trees"] = df_all["f_tree_mean"] - df_all["City_key"].map(trees_ftree0)


# ============================================================
# (A) RANGE-HOLDOUT EXTRAPOLATION EXPERIMENTS
# ============================================================
def run_range_holdout(train_states, holdout_states, tag):
    print(f"\n[A/{tag}] Train on {sorted(train_states)}, hold out {sorted(holdout_states)} ...")
    df_train = df_all[df_all.state_label.isin(train_states)].reset_index(drop=True)
    df_hold  = df_all[df_all.state_label.isin(holdout_states)].reset_index(drop=True)

    model, feature_cols = fit_model(df_train, n_estimators=N_EST)
    pred = predict_with(model, feature_cols, df_hold)
    obs  = df_hold["y"].to_numpy(float)

    # Per-city extrapolation distance: held-out row's f_tree_mean minus
    # THAT city's own max f_tree_mean actually present in df_train.
    city_train_max = df_train.groupby("City_key")["f_tree_mean"].max()
    extrap_dist = df_hold["f_tree_mean"].to_numpy(float) - df_hold["City_key"].map(city_train_max).to_numpy(float)

    out = pd.DataFrame({
        "experiment": tag,
        "City": df_hold["City"].values,
        "City_key": df_hold["City_key"].values,
        "state_label": df_hold["state_label"].values,
        "DayIndex": df_hold["DayIndex"].values,
        "f_tree_mean": df_hold["f_tree_mean"].values,
        "extrap_dist": extrap_dist,
        "ECAC_obs": obs,
        "ECAC_pred": pred,
        "error_pred_minus_obs": pred - obs,
    })

    daily_metrics = {
        "rmse": rmse_np(pred, obs), "mae": mae_np(pred, obs),
        "bias": bias_np(pred, obs), "r2": r2_np(pred, obs),
        "n_rows": int(len(out)),
    }
    print(f"[A/{tag}] DAILY  RMSE={daily_metrics['rmse']:.4g}  MAE={daily_metrics['mae']:.4g}  "
          f"BIAS(pred-obs)={daily_metrics['bias']:+.4g}  R2={daily_metrics['r2']:.4g}  n={daily_metrics['n_rows']}")

    citymean = (
        out.groupby(["City_key", "state_label"], as_index=False)
           .agg(City=("City", "first"),
                extrap_dist=("extrap_dist", "mean"),
                ECAC_obs_summer_mean=("ECAC_obs", "mean"),
                ECAC_pred_summer_mean=("ECAC_pred", "mean"))
    )
    citymean_metrics = {
        "rmse": rmse_np(citymean.ECAC_pred_summer_mean, citymean.ECAC_obs_summer_mean),
        "mae": mae_np(citymean.ECAC_pred_summer_mean, citymean.ECAC_obs_summer_mean),
        "bias": bias_np(citymean.ECAC_pred_summer_mean, citymean.ECAC_obs_summer_mean),
        "r2": r2_np(citymean.ECAC_pred_summer_mean, citymean.ECAC_obs_summer_mean),
        "n_city_states": int(len(citymean)),
    }
    print(f"[A/{tag}] CITYMEAN RMSE={citymean_metrics['rmse']:.4g}  MAE={citymean_metrics['mae']:.4g}  "
          f"BIAS(pred-obs)={citymean_metrics['bias']:+.4g}  R2={citymean_metrics['r2']:.4g}  "
          f"n_city_states={citymean_metrics['n_city_states']}")

    out.to_csv(os.path.join(OUT_DIR, f"ExtrapHoldout_{tag}_daily.csv"), index=False)
    citymean.to_csv(os.path.join(OUT_DIR, f"ExtrapHoldout_{tag}_citymean.csv"), index=False)

    return out, daily_metrics, citymean_metrics, model, feature_cols


holdA1, metricsA1_daily, metricsA1_city, modelA1, featA1 = run_range_holdout(
    train_states={"notree", "trees"}, holdout_states={"plus10", "plus20"}, tag="A1_trainNoTreeTrees")

holdA2, metricsA2_daily, metricsA2_city, modelA2, featA2 = run_range_holdout(
    train_states={"notree", "trees", "plus10"}, holdout_states={"plus20"}, tag="A2_trainUpToPlus10")

pooled = pd.concat([holdA1, holdA2], axis=0, ignore_index=True)
pooled = pooled[pooled.extrap_dist > 0].copy()  # keep genuine extrapolation points only
pooled.to_csv(os.path.join(OUT_DIR, "ExtrapHoldout_Pooled_daily.csv"), index=False)

# Empirical error-vs-distance trend (simple OLS on |error| and on signed
# error, both daily-level). This is the number to look at when judging a
# further step out to +30%.
def ols_trend(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(len(x))}
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2_np(yhat, y), "n": int(len(x))}

trend_abs_error = ols_trend(pooled["extrap_dist"], pooled["error_pred_minus_obs"].abs())
trend_signed_error = ols_trend(pooled["extrap_dist"], pooled["error_pred_minus_obs"])

print("\n[A/TREND] |error| vs extrapolation distance (pooled A1+A2, daily-level):")
print(f"  slope={trend_abs_error['slope']:.4g} (Wh/m^2/d per +1.0 f_tree_mean unit of extra distance), "
      f"intercept={trend_abs_error['intercept']:.4g}, fit R2={trend_abs_error['r2']:.4g}, n={trend_abs_error['n']}")
print("[A/TREND] signed error (pred-obs) vs extrapolation distance:")
print(f"  slope={trend_signed_error['slope']:.4g}, intercept={trend_signed_error['intercept']:.4g}, "
      f"fit R2={trend_signed_error['r2']:.4g}, n={trend_signed_error['n']}")
if trend_signed_error["slope"] < 0:
    print("[A/TREND] Negative slope on SIGNED error -> as extrapolation distance grows, pred-obs falls, "
          "i.e. the model increasingly OVER-predicts ECAC (UNDER-states the cooling benefit) the further "
          "past its training ceiling it's pushed -- the plateau/understatement failure mode.")
elif trend_signed_error["slope"] > 0:
    print("[A/TREND] Positive slope on SIGNED error -> the model increasingly UNDER-predicts ECAC "
          "(OVER-states the cooling benefit) with distance -- the opposite, and arguably more dangerous, "
          "failure mode for an 'offsetable' claim. Worth double-checking before trusting a +30% sweep.")

with open(os.path.join(OUT_DIR, "ExtrapHoldout_ErrorTrend_Summary.json"), "w") as f:
    json.dump({
        "A1_trainNoTreeTrees_daily": metricsA1_daily, "A1_trainNoTreeTrees_citymean": metricsA1_city,
        "A2_trainUpToPlus10_daily": metricsA2_daily, "A2_trainUpToPlus10_citymean": metricsA2_city,
        "pooled_trend_abs_error_vs_extrap_dist": trend_abs_error,
        "pooled_trend_signed_error_vs_extrap_dist": trend_signed_error,
        "note": "extrap_dist is in the same units as f_tree_mean (fraction, 0-1), i.e. slope is per "
                "1.0 = 100 percentage points; multiply by 0.01 for a 'per percentage point' slope.",
    }, f, indent=2)


# ============================================================
# (B) LEAVE-ONE-OUT ON GENUINELY-UNCAPPED CITIES (secondary check)
# ============================================================
print("\n[B] Leave-one-out CV on cities that genuinely reached the +20% target ...")
plus20 = df_all[df_all.state_label == "plus20"].copy()
plus20["nominal_target"] = plus20["City_key"].map(trees_ftree0) + 0.20
plus20["reached_frac"] = plus20["f_tree_mean"] / plus20["nominal_target"]

uncapped_keys = sorted(plus20.loc[plus20.reached_frac >= REACHED_TARGET_FRAC, "City_key"].unique())
print(f"[B] {len(uncapped_keys)}/{plus20['City_key'].nunique()} cities reached >= "
      f"{REACHED_TARGET_FRAC:.0%} of their nominal +20% target and qualify for this test.")

if len(uncapped_keys) < N_FOLDS_EXPERIMENT_B:
    print("[B] Too few uncapped cities for the requested fold count -- skipping Experiment B.")
else:
    uncapped_plus20 = plus20[plus20.City_key.isin(uncapped_keys)].reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS_EXPERIMENT_B)
    groups = uncapped_plus20["City_key"].values
    rows_B = []
    for fold, (_, held_idx) in enumerate(gkf.split(uncapped_plus20, groups=groups)):
        held_keys = set(uncapped_plus20.iloc[held_idx]["City_key"].unique())
        df_train_B = df_all[~((df_all.state_label == "plus20") & (df_all.City_key.isin(held_keys)))].reset_index(drop=True)
        df_hold_B = uncapped_plus20.iloc[held_idx].reset_index(drop=True)

        model_B, feat_B = fit_model(df_train_B, n_estimators=N_EST)
        pred_B = predict_with(model_B, feat_B, df_hold_B)
        obs_B = df_hold_B["y"].to_numpy(float)

        rows_B.append(pd.DataFrame({
            "fold": fold, "City": df_hold_B["City"].values, "City_key": df_hold_B["City_key"].values,
            "DayIndex": df_hold_B["DayIndex"].values, "f_tree_mean": df_hold_B["f_tree_mean"].values,
            "ECAC_obs": obs_B, "ECAC_pred": pred_B, "error_pred_minus_obs": pred_B - obs_B,
        }))
        print(f"[B] fold={fold}: {len(held_keys)} held-out cities, "
              f"RMSE={rmse_np(pred_B, obs_B):.4g}, BIAS={bias_np(pred_B, obs_B):+.4g}")

    df_B = pd.concat(rows_B, axis=0, ignore_index=True)
    df_B.to_csv(os.path.join(OUT_DIR, "ExtrapValidation_B_LeaveOneOutUncapped_daily.csv"), index=False)
    print(f"[B] OVERALL: RMSE={rmse_np(df_B.ECAC_pred, df_B.ECAC_obs):.4g}  "
          f"MAE={mae_np(df_B.ECAC_pred, df_B.ECAC_obs):.4g}  "
          f"BIAS={bias_np(df_B.ECAC_pred, df_B.ECAC_obs):+.4g}  "
          f"R2={r2_np(df_B.ECAC_pred, df_B.ECAC_obs):.4g}")


# ============================================================
# (C) DIAGNOSTIC PLOTS
# ============================================================
print("\n[C] Building diagnostic plots ...")

# C1: predicted vs observed, held-out extrapolation rows, colored by distance
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, (holdout_df, tag) in zip(axes, [(holdA1, "A1: train NoTree+Trees"), (holdA2, "A2: train up to Plus10")]):
    hd = holdout_df[holdout_df.extrap_dist > 0]
    sc = ax.scatter(hd.ECAC_obs, hd.ECAC_pred, c=hd.extrap_dist, cmap="viridis", s=14, alpha=0.7)
    lims = [min(hd.ECAC_obs.min(), hd.ECAC_pred.min()), max(hd.ECAC_obs.max(), hd.ECAC_pred.max())]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlabel("Observed ECAC (true, held out)")
    ax.set_ylabel("Predicted ECAC (extrapolated)")
    ax.set_title(tag)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.colorbar(sc, ax=ax, label="extrap. distance (f_tree_mean units)")
fig.suptitle("Range-holdout extrapolation: predicted vs. observed ECAC beyond the training f_tree ceiling")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Extrap_PredVsObs.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# C2: signed error vs extrapolation distance, pooled, with the fitted trend
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(pooled.extrap_dist, pooled.error_pred_minus_obs, s=10, alpha=0.35, color="#2166ac")
xs = np.linspace(pooled.extrap_dist.min(), pooled.extrap_dist.max(), 50)
ax.plot(xs, trend_signed_error["slope"] * xs + trend_signed_error["intercept"], color="#b2182b", lw=2,
        label=f"OLS trend (slope={trend_signed_error['slope']:.3g})")
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xlabel("Extrapolation distance beyond this city's training f_tree ceiling")
ax.set_ylabel("Predicted - Observed ECAC  (>0 = model UNDER-states cooling benefit)")
ax.set_title("Error growth with extrapolation distance (pooled A1+A2, daily)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Extrap_ErrorVsDistance_Pooled.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# C3: per-city sweep curves, out to +30pp, for a handful of representative
# cities -- overlay the REAL held-out Plus10/Plus20 points.
rng = np.random.default_rng(RANDOM_SEED)
# One row per city (reached_frac is constant across a city's days, but
# plus20 has one row per city-DAY -- .set_index() without collapsing
# first would give a non-unique index and break the later .get(ck) calls).
cap_status = plus20.groupby("City_key")["reached_frac"].mean()
uncapped_examples = list(cap_status[cap_status >= REACHED_TARGET_FRAC].sample(
    min(3, (cap_status >= REACHED_TARGET_FRAC).sum()), random_state=RANDOM_SEED).index)
capped_examples = list(cap_status[cap_status < 0.7].sample(
    min(3, (cap_status < 0.7).sum()), random_state=RANDOM_SEED).index)
example_keys = uncapped_examples + capped_examples

if example_keys:
    ncols = 3
    nrows = int(np.ceil(len(example_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, ck in enumerate(example_keys):
        ax = axes[i // ncols][i % ncols]
        trees_row = df_all[(df_all.City_key == ck) & (df_all.state_label == "trees")].iloc[0]
        f0 = trees_row["f_tree_mean"]
        sweep_deltas = np.linspace(0, SWEEP_MAX_DELTA, 61)
        sweep_df = pd.DataFrame([trees_row[["City"] + NUM_VARS] for _ in sweep_deltas])
        sweep_df = sweep_df.reset_index(drop=True)
        sweep_df["f_tree_mean"] = f0 + sweep_deltas
        # LAI scaled proportionally to f_tree_mean's own relative change vs baseline,
        # matching how Plus10/Plus20 LAI co-varies with f_tree in the real data --
        # a simple assumption, adjust if you have the real LAI-vs-ftree relationship.
        lai0 = trees_row["LAI_TC_mean"]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(f0 > 0, sweep_df["f_tree_mean"] / f0, 1.0)
        sweep_df["LAI_TC_mean"] = lai0 * scale

        predA1 = predict_with(modelA1, featA1, sweep_df)
        predA2 = predict_with(modelA2, featA2, sweep_df)

        ax.plot(sweep_deltas, predA1, color="#b2182b", lw=1.8, label="model A1 (trained NoTree+Trees only)")
        ax.plot(sweep_deltas, predA2, color="#2166ac", lw=1.8, label="model A2 (trained up to Plus10)")

        real = df_all[(df_all.City_key == ck) & (df_all.state_label.isin(["trees", "plus10", "plus20"]))]
        real_delta = real["f_tree_mean"] - f0
        real_grouped = pd.DataFrame({"delta": real_delta, "y": real["y"]}).groupby("delta", as_index=False).mean()
        ax.scatter(real_grouped.delta, real_grouped.y, color="k", zorder=5, s=40,
                   label="real simulated ECAC (Trees/Plus10/Plus20)")

        reached = cap_status.get(ck, np.nan)
        ax.axvline(0.10, color="0.6", lw=0.6, ls=":")
        ax.axvline(0.20, color="0.6", lw=0.6, ls=":")
        ax.set_title(f"{trees_row['City']}  (Plus20 reached {reached:.0%} of target)" if np.isfinite(reached)
                     else f"{trees_row['City']}", fontsize=10)
        ax.set_xlabel("Δ f_tree_mean above this city's Trees baseline")
        ax.set_ylabel("Predicted / observed daily ECAC")
        if i == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper right")

    for j in range(len(example_keys), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Per-city extrapolation sweep: does the reduced-training-range model reach the real held-out point?")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Extrap_PerCity_SweepCurves_Examples.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

print(f"\nDONE. All diagnostics saved to: {OUT_DIR}")
print("Read, in order: ExtrapHoldout_ErrorTrend_Summary.json (the headline numbers), then")
print("Extrap_ErrorVsDistance_Pooled.png, then Extrap_PerCity_SweepCurves_Examples.png.")
