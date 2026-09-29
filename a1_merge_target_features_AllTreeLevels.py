
"""
merge_target_features_AllTreeLevels.py

Merges the two AllTreeLevels pipeline outputs into ONE stacked,
training-ready table covering all four tree states (trees / notree /
plus10 / plus20):

  - TARGET_MAT  : SummerDaily_ECAC_IndexBased_100cities_TROPwarmest3mo_WEIGHTED_AllTreeLevels_PYFRIENDLY.mat
                  (variable ResultsPY -- run
                  Convert_Results_ToPYFriendly_AllTreeLevels.m FIRST to
                  produce this from the raw AllTreeLevels.mat; see that
                  script's header for why the conversion is necessary).
                  Daily_Trees / Daily_NoTree / Daily_Plus10 / Daily_Plus20
                  are cell arrays (one cell per city) of [nDays x 3]
                  matrices; column 0 = BEM_AC_FloorArea (ECAC, the
                  regression target), matching TARGET_COL below.

  - FEATURE_CSV : UTC_BEM_Emulator_Input_DailySummer_AllTreeLevels_WEIGHTED.csv
                  Long table with a TreeState column in
                  {"trees","notree","plus10","plus20"}.

This exactly follows the merge pattern already used in
XGB_model_training.py for the 2-state (Trees/NoTree) case --
df_feat.merge(df_target, on=["City","DayIndex"], how="inner"), then a
>=95%-kept QC assert -- just generalized from 2 hardcoded blocks to a
loop over all 4 states, each tagged with a "state_label" column.

Output: STACKED_AllTreeLevels_n<rows>.csv in OUT_DIR, ready to be read
directly by a training script (no merge step needed there anymore).
"""
import os
import re
import unicodedata

import h5py
import numpy as np
import pandas as pd

# =========================
# PATHS -- EDIT THESE
# =========================
TARGET_MAT = r"E:\UTC_BEM_100cities_loop_TEST_SEED\SummerDaily_ECAC_100cities_AllTreeLevels_TEST_SEED.mat"
FEATURE_CSV = r"E:\UTC_BEM_100cities_loop\UTC_BEM_Emulator_Input_DailySummer_AllTreeLevels_WEIGHTED.csv"
OUT_DIR = r"E:\Tree_paper_Code"

MIN_PCT_KEEP = 95.0  # same QC threshold XGB_model_training.py uses

# TreeState (feature CSV) -> Results/ResultsPY field (target mat)
STATE_TO_TARGET_FIELD = {
    "trees": "Daily_Trees",
    "notree": "Daily_NoTree",
    "plus10": "Daily_Plus10",
    "plus20": "Daily_Plus20",
}
TARGET_COL = 0  # 0 = BEM_AC_FloorArea (ECAC); 1 = _H_ ; 2 = _LE_ -- same convention as XGB_model_training.py

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
# HELPERS (same logic as XGB_model_training.py / e6-e8 scripts)
# ============================================================
def normalize_city_key(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s)


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
    """Same as XGB_model_training.py's function of the same name,
    generalized to whichever Daily_* field is requested."""
    rows = []
    with h5py.File(mat_path, "r") as f:
        if "ResultsPY" not in f:
            raise KeyError(
                f"{mat_path} has no top-level 'ResultsPY'. Did you run "
                f"Convert_Results_ToPYFriendly_AllTreeLevels.m first? "
                f"Top-level keys found: {list(f.keys())}"
            )
        g = f["ResultsPY"]
        if source not in g:
            raise KeyError(f"'{source}' not found under ResultsPY. Available: {list(g.keys())}")

        city_cell = g["City"]
        data_cell = g[source]
        n_city = city_cell.shape[1]

        n_empty = 0
        for i in range(n_city):
            city = _read_matlab_string(f, city_cell[0, i])
            raw = f[data_cell[0, i]][()]
            arr = np.squeeze(np.array(raw))

            if arr.size == 0:
                n_empty += 1
                continue

            if arr.ndim == 1:
                # a degenerate 1-day city squeezes to 1D; treat it as the
                # 3-column row it actually is
                arr = arr.reshape(1, -1)
            if arr.shape[0] == 3 and arr.shape[1] != 3:
                arr = arr.T

            yv = arr[:, int(col)].astype(float)
            for d in range(len(yv)):
                rows.append((city, d + 1, float(yv[d])))

        if n_empty:
            print(
                f"[WARN] {source}: {n_empty}/{n_city} cities had an empty target "
                f"cell (no usable +10%/+20% data for that city) and were skipped."
            )

    return pd.DataFrame(rows, columns=["City", "DayIndex", "y"])


# ============================================================
# (1) LOAD FEATURES
# ============================================================
print("[1/3] Loading FEATURE_CSV ...")
df_feat_all = pd.read_csv(FEATURE_CSV, sep=None, engine="python")
df_feat_all.columns = [str(c).strip() for c in df_feat_all.columns]
df_feat_all["City"] = df_feat_all["City"].astype(str)
df_feat_all["TreeState"] = df_feat_all["TreeState"].astype(str).str.strip().str.lower()

df_feat_all["DayIndex"] = pd.to_numeric(df_feat_all["DayIndex"], errors="coerce")
df_feat_all = df_feat_all.dropna(subset=["City", "DayIndex", "TreeState"]).copy()
df_feat_all["DayIndex"] = df_feat_all["DayIndex"].astype(int)
df_feat_all["City_key"] = df_feat_all["City"].apply(normalize_city_key)

missing_feat = [c for c in NUM_VARS if c not in df_feat_all.columns]
if missing_feat:
    raise ValueError(f"FEATURE_CSV missing NUM_VARS: {missing_feat}")
for c in NUM_VARS:
    df_feat_all[c] = pd.to_numeric(df_feat_all[c], errors="coerce")

print(f"    {len(df_feat_all)} feature rows total. Rows by TreeState:")
print(df_feat_all["TreeState"].value_counts().to_string())

unknown_states = sorted(set(df_feat_all["TreeState"]) - set(STATE_TO_TARGET_FIELD))
if unknown_states:
    raise ValueError(f"FEATURE_CSV has TreeState values with no target mapping: {unknown_states}")

# ============================================================
# (2) LOAD TARGETS + MERGE, ONE STATE AT A TIME
# ============================================================
print("\n[2/3] Loading targets from TARGET_MAT and merging per TreeState ...")
blocks = []
for state, field in STATE_TO_TARGET_FIELD.items():
    df_feat = df_feat_all[df_feat_all["TreeState"] == state].copy()
    if df_feat.empty:
        print(f"[WARN] No feature rows for TreeState='{state}'; skipping.")
        continue

    df_target = load_daily_target_from_ResultsPY(TARGET_MAT, source=field, col=TARGET_COL)

    df_block = df_feat.merge(df_target, on=["City", "DayIndex"], how="inner")
    df_block = df_block.dropna(subset=["y"]).reset_index(drop=True)
    df_block["state_label"] = state

    n_feat = len(df_feat)
    n_merged = len(df_block)
    pct_kept = 100.0 * n_merged / n_feat if n_feat else 0.0
    print(f"    [{state:>6s}] {n_feat} feat rows -> {n_merged} merged rows ({pct_kept:.1f}% kept)")

    if pct_kept < MIN_PCT_KEEP:
        feat_cities = set(df_feat["City"].unique())
        target_cities = set(df_target["City"].unique())
        missing_in_target = sorted(feat_cities - target_cities)
        print(f"    [DIAG] Cities in features but not in target ('{state}'): {missing_in_target}")
        raise AssertionError(
            f"'{state}' merge kept only {pct_kept:.1f}% of feature rows. "
            f"Expected >={MIN_PCT_KEEP:.0f}%. Check City name alignment and DayIndex coverage."
        )

    blocks.append(df_block)

if not blocks:
    raise RuntimeError("No TreeState blocks produced any merged rows -- nothing to save.")

# ============================================================
# (3) STACK + SAVE
# ============================================================
print("\n[3/3] Stacking all TreeState blocks and saving ...")
df_stacked = pd.concat(blocks, ignore_index=True)

print(f"    Total stacked rows: {len(df_stacked)}")
print(df_stacked.groupby("state_label").size().to_string())

out_csv = os.path.join(OUT_DIR, f"STACKED_AllTreeLevels_n{len(df_stacked)}.csv")
df_stacked.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print("\nColumns:", list(df_stacked.columns))
print("\nSample rows:")
print(df_stacked.sample(min(8, len(df_stacked)), random_state=0).to_string())
