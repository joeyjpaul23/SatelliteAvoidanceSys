"""
Method 2: Gradient boosting / ensemble (reproducing Metz & Dart's approach
direction, TU Darmstadt 2020 thesis / Metz, Letizia & Simon, 8th ECSD 2021).

Design, following the source paper as closely as the contract's information
constraints allow:
  1. "Event aggregation setup" feature engineering: condense each event's
     CDM history (>=2 days before TCA only) into one row per event via
     last/min/max/mean of every numeric field, plus CDM count and time span.
  2. Train an ensemble regressor per chaser position-uncertainty component
     (c_sigma_r, c_sigma_t, c_sigma_n) -- NOT risk directly. This mirrors the
     paper's core finding: predicting position uncertainty (which risk is
     derived from) rather than predicting risk end-to-end.
  3. Convert predicted covariance + last-known geometry into a risk value via
     the 2D Pc method (pc_calculator.py), following the paper's own approach
     for filling in what the ensemble doesn't predict: correlation factors
     and relative position/target covariance all use the "naive" (last
     recorded) value, exactly as Section 4.1 of the source paper describes.

See NOTES.md for the full list of documented deviations from the source
paper (regressor family, frame-combination simplification, HBR defaults).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from pc_calculator import compute_risk

CUTOFF = 2.0
TARGET_COLS = ["c_sigma_r", "c_sigma_t", "c_sigma_n"]

NON_FEATURE_COLS = {"event_id", "mission_id", "c_object_type"}

SIGMA_COLS = [
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n",
    "t_sigma_rdot", "t_sigma_tdot", "t_sigma_ndot",
    "c_sigma_rdot", "c_sigma_tdot", "c_sigma_ndot",
]
DET_COLS = ["t_position_covariance_det", "c_position_covariance_det"]
SIGMA_SANITY_BOUND = 1e5   # meters; legitimate values are ~1e1-1e4, sentinel is ~6.4e7
DET_SANITY_BOUND = 1e15    # legitimate values are ~1e1-1e10, sentinel is ~6.7e46


def sanitize_raw(df):
    """Replace known data-quality sentinels (degenerate-covariance error codes,
    e.g. an exact 6.7e46 determinant repeated across ~1.5% of rows) with NaN,
    rather than treating them as real extreme values. Matches the source
    paper's data-prep step of removing "physically meaningless values"
    (Section 2.1), applied at the raw-CDM-row level, before aggregation, so
    a bad row never silently drags a legitimate row's min/max/mean off
    scale, and no event is ever dropped outright (unlike the paper, which
    can afford to drop whole CDM rows since it isn't bound to producing a
    prediction for every test event_id)."""
    df = df.copy()
    for c in SIGMA_COLS + DET_COLS:
        if c not in df.columns:
            continue
        bound = SIGMA_SANITY_BOUND if c in SIGMA_COLS else DET_SANITY_BOUND
        df.loc[df[c].abs() > bound, c] = np.nan
    return df


def numeric_feature_columns(df):
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def aggregate_events(df, numeric_cols):
    """Event-aggregation setup: one row per event_id, last/min/max/mean of
    every numeric CDM field computed over rows with time_to_tca >= CUTOFF."""
    filtered = df[df.time_to_tca >= CUTOFF].copy()

    idx_last = filtered.groupby("event_id")["time_to_tca"].idxmin()
    last_rows = filtered.loc[idx_last].set_index("event_id")

    grouped = filtered.groupby("event_id")
    agg = grouped[numeric_cols].agg(["min", "max", "mean"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]

    last_features = last_rows[numeric_cols].add_suffix("_last")
    meta = pd.DataFrame(
        {
            "num_cdms": grouped.size(),
            "obs_span": grouped["time_to_tca"].max() - grouped["time_to_tca"].min(),
            "c_object_type": last_rows["c_object_type"],
        }
    )

    out = pd.concat([last_features, agg, meta], axis=1)
    out.index.name = "event_id"
    return out.reset_index()


def build_training_labels(train_df_full):
    """True 'final' chaser sigma per event: the global-minimum-time_to_tca
    row (closest available CDM to TCA), from the UNFILTERED training data."""
    idx_final = train_df_full.groupby("event_id")["time_to_tca"].idxmin()
    final_rows = train_df_full.loc[idx_final].set_index("event_id")
    labels = final_rows[TARGET_COLS + ["time_to_tca"]].rename(
        columns={"time_to_tca": "label_time_to_tca"}
    )
    return labels


def one_hot_align(train_feat, test_feat):
    combined = pd.concat(
        [train_feat.assign(_split="train"), test_feat.assign(_split="test")],
        axis=0,
        ignore_index=True,
    )
    combined = pd.get_dummies(combined, columns=["c_object_type"], dummy_na=True)
    train_out = combined[combined._split == "train"].drop(columns="_split")
    test_out = combined[combined._split == "test"].drop(columns="_split")
    return train_out.reset_index(drop=True), test_out.reset_index(drop=True)


def main():
    train_full = sanitize_raw(pd.read_csv("../../data/extracted/train_data.csv"))
    test_full = sanitize_raw(pd.read_csv("../../data/test_data.csv"))

    numeric_cols = numeric_feature_columns(test_full)

    train_feat = aggregate_events(train_full, numeric_cols)
    test_feat = aggregate_events(test_full, numeric_cols)

    labels = build_training_labels(train_full)

    train_merged = train_feat.merge(labels, on="event_id", how="inner")
    # Only keep training events where a genuine post-cutoff CDM exists to
    # serve as a real label (not the same row already used as input).
    train_merged = train_merged[train_merged.label_time_to_tca < CUTOFF]
    # sanitize_raw() can turn a sentinel-value label into NaN -- drop those
    # events rather than train on a fabricated label
    train_merged = train_merged.dropna(subset=TARGET_COLS)

    print(f"training events available: {len(train_feat)}")
    print(f"training events with usable post-cutoff label: {len(train_merged)}")
    print(f"test events: {len(test_feat)}")

    train_X_raw = train_merged.drop(columns=TARGET_COLS + ["label_time_to_tca", "event_id"])
    test_X_raw = test_feat.drop(columns=["event_id"])

    train_X, test_X = one_hot_align(train_X_raw, test_X_raw)

    # median-impute any NaNs (from events with sparse CDM history), fit on train only
    medians = train_X.median(numeric_only=True)
    train_X = train_X.fillna(medians)
    test_X = test_X.fillna(medians)

    # Source paper removes rows outside the 5th-95th percentile / physically
    # meaningless values before training (Section 2.1). We've already
    # aggregated to event-level, so row removal isn't natural here -- instead
    # we winsorize each feature to its train-set 0.5/99.5 percentile range,
    # which serves the same purpose (stop pathological outlier CDMs, e.g. a
    # ~6.7e46 covariance determinant found in this data, from dominating tree
    # splits or overflowing float32 during training).
    numeric_feature_cols = train_X.select_dtypes(include=[np.number]).columns
    lo = train_X[numeric_feature_cols].quantile(0.005)
    hi = train_X[numeric_feature_cols].quantile(0.995)
    train_X[numeric_feature_cols] = train_X[numeric_feature_cols].clip(lo, hi, axis=1)
    test_X[numeric_feature_cols] = test_X[numeric_feature_cols].clip(lo, hi, axis=1)

    models = {}
    predictions = {}
    for target in TARGET_COLS:
        y = train_merged[target].values
        # same winsorization rationale as the feature matrix, applied to the
        # regression target so a handful of pathological labels don't
        # dominate a squared-error loss
        y_lo, y_hi = np.percentile(y, [0.5, 99.5])
        y = np.clip(y, y_lo, y_hi)
        model = GradientBoostingRegressor(n_estimators=100, random_state=0)
        model.fit(train_X, y)
        models[target] = model
        predictions[target] = model.predict(test_X)
        print(f"trained GBM for {target}, train R2={model.score(train_X, y):.3f}")

    # Assemble final risk per test event using last-known geometry/correlations
    # + GBM-predicted chaser sigmas, per pc_calculator's compute_risk.
    # Sanitizing sentinels can leave NaNs on the single closest-to-cutoff row,
    # so forward-fill each event's most recent *valid* value toward the
    # cutoff (rather than taking the raw last row verbatim), falling back to
    # the training-set global median only if an event has no valid value at
    # all for a given field.
    geometry_cols = [
        "t_sigma_r", "t_sigma_t", "t_sigma_n", "t_ct_r", "t_cn_r", "t_cn_t",
        "c_ct_r", "c_cn_r", "c_cn_t",
        "relative_position_r", "relative_position_t", "relative_position_n",
        "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
    ]
    test_sorted = test_full[test_full.time_to_tca >= CUTOFF].sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    )
    test_sorted[geometry_cols] = test_sorted.groupby("event_id")[geometry_cols].ffill()
    test_last = test_sorted.loc[
        test_sorted.groupby("event_id")["time_to_tca"].idxmin()
    ].set_index("event_id")

    global_medians = train_full[geometry_cols].median()
    test_last[geometry_cols] = test_last[geometry_cols].fillna(global_medians)
    test_last["c_object_type"] = test_last["c_object_type"].fillna("UNKNOWN")

    results = []
    for i, event_id in enumerate(test_feat["event_id"]):
        row = test_last.loc[event_id]
        risk = compute_risk(
            t_sigma_r=row.t_sigma_r, t_sigma_t=row.t_sigma_t, t_sigma_n=row.t_sigma_n,
            t_ct_r=row.t_ct_r, t_cn_r=row.t_cn_r, t_cn_t=row.t_cn_t,
            c_sigma_r=predictions["c_sigma_r"][i],
            c_sigma_t=predictions["c_sigma_t"][i],
            c_sigma_n=predictions["c_sigma_n"][i],
            c_ct_r=row.c_ct_r, c_cn_r=row.c_cn_r, c_cn_t=row.c_cn_t,
            rel_pos_rtn=[row.relative_position_r, row.relative_position_t, row.relative_position_n],
            rel_vel_rtn=[row.relative_velocity_r, row.relative_velocity_t, row.relative_velocity_n],
            c_object_type=row.c_object_type,
        )
        results.append((event_id, risk))

    out = pd.DataFrame(results, columns=["event_id", "predicted_risk"]).sort_values("event_id")
    out.to_csv("predictions.csv", index=False)
    print(f"wrote {len(out)} predictions to predictions.csv")
    print(out.describe())


if __name__ == "__main__":
    main()
