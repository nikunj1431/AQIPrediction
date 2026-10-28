import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene", "Xylene"]
ALIASES = {"datetime": "Date", "date": "Date", "aqi": "AQI", "aqi value": "AQI",
           "aqi_value": "AQI", "aqi_bucket": "AQI_Bucket",
           "air quality status": "AQI_Bucket", "pm2.5": "PM2.5", "pm10": "PM10",
           "no": "NO", "no2": "NO2", "nox": "NOx", "nh3": "NH3", "co": "CO",
           "so2": "SO2", "o3": "O3", "benzene": "Benzene", "toluene": "Toluene",
           "xylene": "Xylene"}
OUT = "decisiontree_output"


def bucket(x):
    if x <= 50: return "Good"
    if x <= 100: return "Satisfactory"
    if x <= 200: return "Moderate"
    if x <= 300: return "Poor"
    if x <= 400: return "Very Poor"
    return "Severe"


def load(path):
    """Load an already-cleaned city CSV and add the time features the tree needs."""
    df = pd.read_csv(path)
    df.columns = [ALIASES.get(c.strip().lower(), c.strip()) for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # hourly files: collapse to daily means so every file is comparable
    df = df.dropna(subset=["Date", "AQI"])
    df["Day"] = df["Date"].dt.normalize()
    num = [c for c in POLLUTANTS + ["AQI"] if c in df.columns]
    df = df.groupby("Day", as_index=False)[num].mean().rename(columns={"Day": "Date"})
    df = df.sort_values("Date").reset_index(drop=True)

    # keep only pollutant columns that actually have data for this city
    pols = [p for p in POLLUTANTS if p in df.columns and df[p].notna().any()]
    for p in pols:                                   # safety net only (clean.py should have done this)
        df[p] = df[p].fillna(df[p].median())

    df["Year"], df["Month"] = df["Date"].dt.year, df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.weekday
    df["Season"] = df["Month"].map(lambda m: 0 if m in (12, 1, 2) else 1 if m in (3, 4, 5)
                                   else 2 if m in (6, 7, 8, 9) else 3)
    df["Is_Winter"] = df["Month"].isin([11, 12, 1, 2]).astype(int)
    df["Is_Festival_Season"] = df["Month"].isin([10, 11]).astype(int)
    df["AQI_Bucket"] = df["AQI"].apply(bucket)
    df["AQI_Next_Day"] = df["AQI"].shift(-1)
    # only keep a next-day label if the next row really is the next calendar day
    gap = df["Date"].shift(-1) - df["Date"]
    df.loc[gap != pd.Timedelta(days=1), "AQI_Next_Day"] = np.nan
    return df, pols


def split(df, frac=0.8):
    cut = df["Date"].iloc[int(len(df) * frac)]       # train on past, test on future
    return df[df["Date"] < cut], df[df["Date"] >= cut]


def run_regression(df, feats, target, name, tag):
    d = df.dropna(subset=[target])
    tr, te = split(d)
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        {"max_depth": [3, 4, 6, 8, 10], "min_samples_leaf": [1, 5, 10, 20]},
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1).fit(tr[feats], tr[target])
    tree = grid.best_estimator_
    pred = tree.predict(te[feats])
    mae = mean_absolute_error(te[target], pred)
    rmse = np.sqrt(mean_squared_error(te[target], pred))
    r2 = r2_score(te[target], pred)
    print(f"\n[{name}] Decision Tree Regressor  (train={len(tr)}, test={len(te)})")
    print(f"  best params: {grid.best_params_}")
    print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

    imp = pd.Series(tree.feature_importances_, index=feats).sort_values()
    plt.figure(figsize=(7, 5)); imp.tail(12).plot.barh(color="#2e9e4f")
    plt.title(f"Feature importance - {name}"); plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}_importance.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 6)); plt.scatter(te[target], pred, s=8, alpha=0.5)
    lim = [0, max(te[target].max(), pred.max())]
    plt.plot(lim, lim, "r--"); plt.xlabel("Actual AQI"); plt.ylabel("Predicted AQI")
    plt.title(f"Actual vs predicted - {name}"); plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}_actual_vs_pred.png", dpi=150); plt.close()

    plt.figure(figsize=(18, 8))
    plot_tree(tree, max_depth=3, feature_names=feats, filled=True, rounded=True, fontsize=8)
    plt.savefig(f"{OUT}/{tag}_tree.png", dpi=150); plt.close()
    return dict(model=name, MAE=mae, RMSE=rmse, R2=r2)


def run_classification(df, feats, tag):
    tr, te = split(df)
    clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight="balanced",
                                 random_state=42).fit(tr[feats], tr["AQI_Bucket"])
    pred = clf.predict(te[feats])
    y = te["AQI_Bucket"]
    print("\n[AQI category] Decision Tree Classifier")
    print(f"  Accuracy ={accuracy_score(y, pred):.3f}")
    print(f"  Precision={precision_score(y, pred, average='weighted', zero_division=0):.3f}")
    print(f"  Recall   ={recall_score(y, pred, average='weighted', zero_division=0):.3f}")
    print(f"  F1       ={f1_score(y, pred, average='weighted', zero_division=0):.3f}")
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, xticks_rotation=45, cmap="Greens")
    plt.tight_layout(); plt.savefig(f"{OUT}/{tag}_confusion_matrix.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*", default=["Chandigarh_day_15to20.csv"])
    ap.add_argument("--no-pollutants", action="store_true",
                    help="use only date features (no pollutant columns)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for path in args.csvs:
        city = os.path.splitext(os.path.basename(path))[0]
        print("=" * 60, f"\n{city}\n" + "=" * 60)
        df, pols = load(path)
        if len(df) < 50:
            print("Too few rows with AQI - skipping."); continue
        print(f"{len(df)} daily rows | {df['Date'].min().date()} -> {df['Date'].max().date()}")
        time_feats = ["Year", "Month", "Weekday", "Season", "Is_Winter", "Is_Festival_Season"]
        feats = time_feats + ([] if args.no_pollutants else pols)

        r = run_regression(df, feats, "AQI", f"{city} same-day AQI", f"{city}_sameday")
        run_classification(df, feats, city)
        r2 = run_regression(df, feats + ["AQI"], "AQI_Next_Day",
                            f"{city} next-day AQI", f"{city}_nextday")
        rows += [dict(city=city, task="same-day", **r), dict(city=city, task="next-day", **r2)]

    summary = pd.DataFrame(rows).drop(columns="model").round(3)
    summary.to_csv(f"{OUT}/decisiontree_summary.csv", index=False)
    print("\n", summary.to_string(index=False))
    print(f"\nPlots and summary saved in ./{OUT}/")


if __name__ == "__main__":
    main()
