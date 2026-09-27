"""
data_preparation.py
===================
Home Credit Default Risk: reusable data cleaning and feature engineering.

Implements the data preparation decisions recorded in the EDA notebook
(notebooks/eda.qmd, "Results" section). Every transformation below carries a
comment naming the EDA decision it implements.

Core design: fit on train, apply to both
----------------------------------------
Any value that depends on the data (medians, caps, bin cutoffs, allowed
category levels) is learned ONCE from the training data by `fit_preparation()`
and stored in a parameter dictionary. `transform()` then applies those stored
values to train and test alike and never computes statistics from the data it
is given. This prevents test-set information from leaking into preparation and
guarantees identical transformations and identical columns.

Usage
-----
From the project root:

    python data_preparation.py

or with explicit paths:

    python data_preparation.py --train data/raw/application_train.csv \
        --test data/raw/application_test.csv --out data/processed

From Python:

    from data_preparation import fit_preparation, transform
    params = fit_preparation(train)
    train_ready = transform(train, params)
    test_ready = transform(test, params)

Inputs:  application_train.csv, application_test.csv (Kaggle, Home Credit)
Outputs: train_prepared.parquet, test_prepared.parquet, prep_params.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# 1. CONSTANTS
#    Fixed rules taken from the EDA. These are NOT learned from data, so they
#    are safe to hard-code and apply identically everywhere.
# ===========================================================================
ID_COL, TARGET_COL = "SK_ID_CURR", "TARGET"

DAYS_EMPLOYED_PLACEHOLDER = 365243          # EDA: 55,374 rows, 99.96% pensioners
CAR_AGE_PLACEHOLDERS = [64, 65]             # EDA: spike of 3,334 cars; median owner age 37
UNDOCUMENTED_LEVELS = {                     # EDA: levels absent from the data dictionary
    "CODE_GENDER": ["XNA"],
    "NAME_FAMILY_STATUS": ["Unknown"],
}
VALID_REGION_RATINGS = [1, 2, 3]            # EDA: dictionary says 1-3; test contains one -1

# EDA: cap extreme counts and the 117M income entry error at the TRAIN 99.9th percentile
COLS_TO_CAP = [
    "AMT_INCOME_TOTAL",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS",
]

# "Days before application" columns converted to positive years (EDA: DAYS_* are negative)
DAYS_TO_YEARS = {
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_EMPLOYED": "EMPLOYED_YEARS",
    "DAYS_REGISTRATION": "YEARS_SINCE_REGISTRATION",
    "DAYS_ID_PUBLISH": "YEARS_SINCE_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE": "YEARS_SINCE_PHONE_CHANGE",
}

EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
BUREAU_REQ_COLS = [
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
]
# EDA Q8: hour/day/week inquiry counts are almost always zero; keep a single total instead
BUREAU_REQ_TO_DROP = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
                      "AMT_REQ_CREDIT_BUREAU_WEEK"]

# Fixed business cutoffs for binned variables (not learned, so identical by construction)
AGE_BAND_EDGES = [0, 25, 35, 45, 55, 65, 120]                        # EDA Q11 age bands
AGE_BAND_LABELS = ["under_25", "25_34", "35_44", "45_54", "55_64", "65_plus"]
LTV_BAND_EDGES = [-np.inf, 0.9999, 1.0001, 1.1, 1.2, 1.3, np.inf]   # EDA Q5 LTV bands
LTV_BAND_LABELS = ["below_1", "exactly_1", "1_0_to_1_1", "1_1_to_1_2", "1_2_to_1_3", "above_1_3"]

RARE_LEVEL_MIN_COUNT = 30     # levels seen fewer times than this in TRAIN become missing
NEAR_CONSTANT_SHARE = 0.999   # drop columns where one value covers 99.9%+ of TRAIN rows
MISSING_LEVEL = "Missing"     # explicit level for missing categories


# ===========================================================================
# 2. HELPERS (row-by-row logic only; nothing here looks across applicants)
# ===========================================================================
def building_columns(columns) -> list:
    """Return the 47 building/housing statistic columns (suffix _AVG, _MODE or _MEDI)."""
    return [c for c in columns if c.endswith(("_AVG", "_MODE", "_MEDI"))]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two columns, returning NaN instead of infinity when the denominator is 0 or missing."""
    return numerator / denominator.replace(0, np.nan)


# ===========================================================================
# 3. CLEANING: fix bad values without ever removing a row
# ===========================================================================
def clean_application(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Recode placeholders, undocumented levels and extreme values.

    Works identically on train and test. Never drops rows: every test applicant
    stands in for a future customer who must still receive a score.
    """
    out = df.copy()   # never modify the caller's data in place

    # EDA decision: DAYS_EMPLOYED = 365243 is a pensioner placeholder.
    # Flag it first (pensioners default at 5.4%, a real signal), then set it to
    # missing so the placeholder cannot distort any mean, scale or ratio.
    out["FLAG_EMPLOYED_PLACEHOLDER"] = (out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_PLACEHOLDER).astype(int)
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_PLACEHOLDER, np.nan)

    # EDA decision: car ages of 64-65 are a second placeholder (flag, then missing)
    is_car_placeholder = out["OWN_CAR_AGE"].isin(CAR_AGE_PLACEHOLDERS)
    out["FLAG_CAR_AGE_PLACEHOLDER"] = is_car_placeholder.astype(int)
    out.loc[is_car_placeholder, "OWN_CAR_AGE"] = np.nan

    # EDA decision: undocumented levels (XNA gender, Unknown family status) become missing.
    # Note (EDA Q11): CODE_GENDER is KEPT in the prepared data only so the modeling stage
    # can test for disparate impact. It must be excluded from model features, because
    # Regulation B prohibits considering sex in credit decisions.
    for col, bad_levels in UNDOCUMENTED_LEVELS.items():
        out[col] = out[col].replace(bad_levels, np.nan)

    # EDA decision: region rating must be 1, 2 or 3; the -1 found in test becomes missing
    out["REGION_RATING_CLIENT_W_CITY"] = out["REGION_RATING_CLIENT_W_CITY"].where(
        out["REGION_RATING_CLIENT_W_CITY"].isin(VALID_REGION_RATINGS))

    # EDA decision: train-only and rare levels (e.g. Maternity leave, 5 rows) become missing,
    # so train and test share one set of categories and unseen test levels cannot break the pipeline
    for col, levels in params["allowed_levels"].items():
        out[col] = out[col].where(out[col].isin(levels))

    # EDA decision: cap extremes at the TRAIN 99.9th percentile. This replaces the EDA note
    # "drop the 117M income row": capping fixes the error without removing an applicant.
    for col, cap in params["caps"].items():
        out[col] = out[col].clip(upper=cap)

    # EDA decision: drop FLAG_MOBIL and document flags with no information.
    # Chosen by a TRAIN-only near-zero-variance rule rather than by inspecting test.
    out = out.drop(columns=params["drop_cols"])

    return out


# ===========================================================================
# 4. FEATURE ENGINEERING (row-by-row, plus bins that use learned cutoffs)
# ===========================================================================
def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing-data flags. MUST run before imputation, or every flag would be 0."""
    out = df.copy()

    # EDA decision (Q2): no credit bureau inquiry record marks a thin-file applicant.
    # Thin-file applicants default at 10.3% vs 7.7%.
    out["FLAG_THIN_FILE"] = out["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int)

    # EDA decision (Q3): default rises with each missing external score
    for col in EXT_COLS:
        out[f"{col}_MISSING"] = out[col].isna().astype(int)
    out["EXT_SOURCE_MISSING_COUNT"] = out[EXT_COLS].isna().sum(axis=1)

    # EDA decision (missingness check): the 47 building columns go missing as a block,
    # and applicants with none of them default at 9.3% vs 7.0%. One share replaces 47 flags.
    bcols = building_columns(df.columns)
    out["BUILDING_INFO_MISSING_SHARE"] = out[bcols].isna().mean(axis=1)

    # EDA decision (missingness check): missing occupation marks a SAFER group (57% pensioners)
    out["FLAG_OCCUPATION_MISSING"] = out["OCCUPATION_TYPE"].isna().astype(int)

    # EDA decision (Q5): goods price is missing only for revolving loans
    out["FLAG_GOODS_PRICE_MISSING"] = out["AMT_GOODS_PRICE"].isna().astype(int)
    return out


def add_engineered_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Unit fixes, financial ratios, summary features, bins and interactions."""
    out = df.copy()

    # EDA decision: convert negative "days before application" to positive years
    for days_col, years_col in DAYS_TO_YEARS.items():
        out[years_col] = -out[days_col] / 365.25
    out = out.drop(columns=list(DAYS_TO_YEARS))

    # EDA decision (Q9): phone changed on the application day behaves as its own category
    out["FLAG_PHONE_CHANGED_ON_APP_DAY"] = (out["YEARS_SINCE_PHONE_CHANGE"] == 0).astype(int)

    # EDA decision (Q4): payment-to-income, the core ability-to-repay ratio (CFPB).
    # Kept as a candidate even though income looks self-reported.
    out["PAYMENT_TO_INCOME"] = safe_divide(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    # Credit-to-income: total debt relative to income, a standard companion ratio
    out["CREDIT_TO_INCOME"] = safe_divide(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    # EDA decision (Q5): loan-to-value; loans 30%+ above goods price default at 12.8%
    out["LOAN_TO_VALUE"] = safe_divide(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])
    # EDA decision (Q6): credit-to-annuity approximates term and carries past pricing
    out["CREDIT_TO_ANNUITY"] = safe_divide(out["AMT_CREDIT"], out["AMT_ANNUITY"])
    # Job tenure as a share of adult life: separates long-tenured workers from young ones
    out["EMPLOYED_TO_AGE"] = safe_divide(out["EMPLOYED_YEARS"], out["AGE_YEARS"])

    # EDA decision (Q3): the mean of available external scores beats any single score (AUC 0.72)
    out["EXT_SOURCE_MEAN"] = out[EXT_COLS].mean(axis=1)

    # EDA decision (Q8): one total of bureau inquiries instead of six sparse counts
    out["TOTAL_BUREAU_INQUIRIES"] = out[BUREAU_REQ_COLS].sum(axis=1, min_count=1)
    out = out.drop(columns=BUREAU_REQ_TO_DROP)

    # EDA decision: the building block repeats 14 measures three ways (_AVG, _MODE, _MEDI).
    # Keep _AVG versions plus the four text _MODE categories and TOTALAREA_MODE (no _AVG exists).
    numeric_dupes = [c for c in building_columns(out.columns)
                     if c.endswith(("_MODE", "_MEDI"))
                     and c != "TOTALAREA_MODE"
                     and pd.api.types.is_numeric_dtype(out[c])]
    out = out.drop(columns=numeric_dupes)

    # Binned variables ---------------------------------------------------------
    # EDA decision (Q11): age bands with fixed cutoffs (age is permitted; older = lower risk)
    out["AGE_BAND"] = pd.cut(out["AGE_YEARS"], AGE_BAND_EDGES, labels=AGE_BAND_LABELS).astype("object")
    # EDA decision (Q5): LTV bands keep the non-linear jump above 1.2
    out["LTV_BAND"] = pd.cut(out["LOAN_TO_VALUE"], LTV_BAND_EDGES, labels=LTV_BAND_LABELS).astype("object")
    # Income quintiles: cutoffs LEARNED FROM TRAIN in fit_preparation(), reused here unchanged
    out["INCOME_QUINTILE"] = pd.cut(out["AMT_INCOME_TOTAL"], params["income_quintile_edges"],
                                    labels=["q1", "q2", "q3", "q4", "q5"],
                                    include_lowest=True).astype("object")

    # Interaction terms --------------------------------------------------------
    # EDA Q2 + Q3: an external score may mean something different when no bureau file exists
    out["EXT_MEAN_X_THIN_FILE"] = out["EXT_SOURCE_MEAN"] * out["FLAG_THIN_FILE"]
    # EDA Q4 + Q7: payment burden weighs differently on a pension than on a wage
    out["PTI_X_PENSIONER"] = out["PAYMENT_TO_INCOME"] * out["FLAG_EMPLOYED_PLACEHOLDER"]

    return out


# ===========================================================================
# 5. IMPUTATION AND ENCODING (uses values learned from train)
# ===========================================================================
def impute_and_encode(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Fill blanks with TRAIN medians, then one-hot encode with TRAIN category levels.

    Runs after add_missing_indicators(), so the fact that a value was missing
    is already preserved in a flag column.
    """
    out = df.copy()

    # Numeric blanks -> TRAIN median (stored in params, never recomputed on test)
    for col, median in params["medians"].items():
        out[col] = out[col].fillna(median)

    # Categorical blanks -> an explicit "Missing" level
    for col in params["encoded_levels"]:
        out[col] = out[col].fillna(MISSING_LEVEL)

    # One-hot encode using the TRAIN level list, so train and test get exactly the
    # same dummy columns even if a level is absent from one file
    dummies = []
    for col, levels in params["encoded_levels"].items():
        cat = pd.Categorical(out[col], categories=levels)
        dummies.append(pd.get_dummies(cat, prefix=col, dtype=int).set_index(out.index))
    out = pd.concat([out.drop(columns=list(params["encoded_levels"]))] + dummies, axis=1)

    # Fix the final column order to the order learned from train
    ordered = [c for c in params["feature_columns"]]
    extra = [TARGET_COL] if TARGET_COL in out.columns else []
    return out[[ID_COL] + ordered + extra]


# ===========================================================================
# 6. FIT AND TRANSFORM (the public interface)
# ===========================================================================
def _pre_impute(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Shared steps before imputation: clean, flag missingness, engineer features."""
    out = clean_application(df, params)
    out = add_missing_indicators(out)
    return add_engineered_features(out, params)


def fit_preparation(train: pd.DataFrame) -> dict:
    """Learn every data-dependent value from the TRAINING data only.

    Learned values: caps, allowed category levels, near-constant columns to drop,
    income-quintile cutoffs, imputation medians, one-hot level lists and the final
    column order. Returns them in a dictionary that transform() reuses unchanged.
    """
    params = {}

    # EDA decision: caps at the TRAIN 99.9th percentile
    params["caps"] = {c: float(train[c].quantile(0.999)) for c in COLS_TO_CAP}

    # EDA decision: keep only category levels seen at least 30 times in TRAIN
    cat_cols = train.select_dtypes(include=["object", "string"]).columns
    params["allowed_levels"] = {
        c: sorted(lvl for lvl, n in train[c].value_counts().items() if n >= RARE_LEVEL_MIN_COUNT)
        for c in cat_cols
    }

    # EDA decision: drop near-constant columns, judged on TRAIN only (near-zero variance)
    numeric = train.drop(columns=[ID_COL, TARGET_COL], errors="ignore").select_dtypes("number")
    top_share = numeric.apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
    params["drop_cols"] = sorted(top_share[top_share >= NEAR_CONSTANT_SHARE].index)

    # Income quintile cutoffs from TRAIN (after capping, so the entry error cannot skew them).
    # Outer edges are opened to +/- infinity so any test income falls into a bin.
    capped_income = train["AMT_INCOME_TOTAL"].clip(upper=params["caps"]["AMT_INCOME_TOTAL"])
    edges = capped_income.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
    edges[0], edges[-1] = -np.inf, np.inf
    params["income_quintile_edges"] = edges

    # Run the shared steps on TRAIN to learn medians, levels and column order
    staged = _pre_impute(train, params)
    feature_frame = staged.drop(columns=[ID_COL, TARGET_COL], errors="ignore")

    # Learn a median for EVERY numeric column, not only those with blanks in train:
    # a column complete in train can still be blank in test (e.g. the -1 region rating
    # recoded to missing), and test must never be left with a gap it cannot fill.
    num_cols = feature_frame.select_dtypes("number").columns
    params["medians"] = {c: float(feature_frame[c].median()) for c in num_cols}

    obj_cols = feature_frame.select_dtypes(exclude="number").columns
    params["encoded_levels"] = {
        c: sorted(feature_frame[c].dropna().astype(str).unique().tolist()) + [MISSING_LEVEL]
        for c in obj_cols
    }

    # Final column order: numeric features, then every train-level dummy column
    dummy_cols = [f"{c}_{lvl}" for c, lvls in params["encoded_levels"].items() for lvl in lvls]
    params["feature_columns"] = list(num_cols) + dummy_cols
    return params


def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply every preparation step to any dataset (train or test) using learned params.

    Never computes a statistic from df. Returns one row per SK_ID_CURR with the
    columns in params['feature_columns'] (plus TARGET when df contains it).
    """
    return impute_and_encode(_pre_impute(df, params), params)


# ===========================================================================
# 7. CONSISTENCY CHECKS
# ===========================================================================
def check_consistency(train_raw, test_raw, train_ready, test_ready) -> dict:
    """Verify the prepared data meets the train/test consistency requirements.

    Raises AssertionError if any check fails; returns a summary dictionary otherwise.
    """
    train_cols = [c for c in train_ready.columns if c != TARGET_COL]
    checks = {
        "identical_columns_except_target": train_cols == list(test_ready.columns),
        "one_row_per_id_train": train_ready[ID_COL].is_unique,
        "one_row_per_id_test": test_ready[ID_COL].is_unique,
        "no_rows_dropped_train": len(train_ready) == len(train_raw),
        "no_rows_dropped_test": len(test_ready) == len(test_raw),
        "no_missing_values_train": int(train_ready[train_cols].isna().sum().sum()) == 0,
        "no_missing_values_test": int(test_ready.isna().sum().sum()) == 0,
        "target_only_in_train": TARGET_COL in train_ready and TARGET_COL not in test_ready,
    }
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f"Consistency checks failed: {failed}"
    checks["train_shape"] = train_ready.shape
    checks["test_shape"] = test_ready.shape
    return checks


# ===========================================================================
# 8. COMMAND-LINE ENTRY POINT
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Prepare Home Credit application data for modeling.")
    parser.add_argument("--train", default="data/raw/application_train.csv")
    parser.add_argument("--test", default="data/raw/application_test.csv")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)

    params = fit_preparation(train_raw)         # learn from TRAIN only
    train_ready = transform(train_raw, params)  # apply to train
    test_ready = transform(test_raw, params)    # apply the same values to test

    checks = check_consistency(train_raw, test_raw, train_ready, test_ready)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ready.to_parquet(out_dir / "train_prepared.parquet", index=False)
    test_ready.to_parquet(out_dir / "test_prepared.parquet", index=False)
    with open(out_dir / "prep_params.json", "w") as f:   # saved so test can be reprocessed later
        json.dump(params, f, indent=2, default=float)

    print("Consistency checks:")
    for name, value in checks.items():
        print(f"  {name}: {value}")
    print(f"Saved outputs to {out_dir}/")


if __name__ == "__main__":
    main()
