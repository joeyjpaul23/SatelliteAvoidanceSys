"""
Method 1: Naive persistence baseline.

Reproduces the mandatory baseline defined in the shared contract: for each
test event_id, predict the final risk (log10 Pc) at TCA as simply the risk
value from the latest available CDM before the 2-day-prior-to-TCA cutoff.

The Kelvins dataset already excludes CDMs within 2 days of TCA (confirmed:
test_data.csv time_to_tca column has min ~2.0003), so "latest available CDM"
is just the row with the smallest time_to_tca for each event_id.

No training happens here and no test labels are read or needed -- this
method uses only the test set's own input features.
"""
import argparse
import pandas as pd


def predict(test_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(test_csv_path)

    required = {"event_id", "time_to_tca", "risk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test data missing required columns: {missing}")

    if df["time_to_tca"].min() < 2.0:
        raise ValueError(
            "found CDM(s) within 2 days of TCA in the input -- this violates "
            "the shared contract's >=2-day cutoff rule and must be filtered "
            "before use, not silently included"
        )

    latest = df.loc[df.groupby("event_id")["time_to_tca"].idxmin()]
    out = latest[["event_id", "risk"]].rename(columns={"risk": "predicted_risk"})
    out = out.sort_values("event_id").reset_index(drop=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="../../data/test_data.csv")
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()

    predictions = predict(args.test_csv)
    predictions.to_csv(args.out, index=False)
    print(f"wrote {len(predictions)} predictions to {args.out}")
    print(predictions.head())
