
import argparse
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score
)

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]
OUT = "decisiontree_output"
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = ["City", "Date", "AQI", "AQI_Bucket"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["City"] = df["City"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["City", "Date"]).copy()

    # Convert duplicate city-date records into daily averages.
    numeric_cols = [c for c in POLLUTANTS + ["AQI"] if c in df.columns]
    df = df.groupby(["City", "Date"], as_index=False)[numeric_cols].mean()

    # Create date-based features.
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.weekday
    df["Season"] = df["Month"].map(
        lambda m: 0 if m in (12, 1, 2)
        else 1 if m in (3, 4, 5)
        else 2 if m in (6, 7, 8, 9)
        else 3
    )
    df["Is_Winter"] = df["Month"].isin([11, 12, 1, 2]).astype(int)
    df["Is_Festival_Season"] = df["Month"].isin([10, 11]).astype(int)

    # Generate next-day AQI separately for every city.
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    df["AQI_Next_Day"] = df.groupby("City")["AQI"].shift(-1)
    df["Next_Date"] = df.groupby("City")["Date"].shift(-1)

    # Keep a next-day target only when the next observation is
    # exactly one calendar day later.
    gap = df["Next_Date"] - df["Date"]
    df.loc[gap != pd.Timedelta(days=1), "AQI_Next_Day"] = np.nan

    # Remove rows without an AQI category.
    df["AQI_Bucket"] = df["AQI_Bucket"].astype("string").str.strip()
    df.loc[df["AQI_Bucket"] == "", "AQI_Bucket"] = pd.NA

    df = df.sort_values(["Date", "City"]).reset_index(drop=True)

    return df


def chronological_split(df, frac=0.8):
    dates = np.sort(df["Date"].dropna().unique())

    if len(dates) < 10:
        raise ValueError("Not enough unique dates for chronological splitting.")

    cut_index = int(len(dates) * frac)
    cut_index = min(max(cut_index, 1), len(dates) - 1)
    cutoff = pd.Timestamp(dates[cut_index])

    train = df[df["Date"] < cutoff].copy()
    test = df[df["Date"] >= cutoff].copy()

    print(f"\nTraining cutoff date: {cutoff.date()}")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")

    return train, test, cutoff


def make_date_cv(df, n_splits=5):
    """Create expanding-window validation folds using complete dates."""
    dates = np.sort(df["Date"].unique())

    if len(dates) <= n_splits:
        raise ValueError("Not enough dates for time-series cross-validation.")

    blocks = np.array_split(dates, n_splits + 1)
    splits = []

    for i in range(1, n_splits + 1):
        train_dates = np.concatenate(blocks[:i])
        val_dates = blocks[i]

        train_idx = np.flatnonzero(df["Date"].isin(train_dates).to_numpy())
        val_idx = np.flatnonzero(df["Date"].isin(val_dates).to_numpy())

        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))

    return splits


def make_pipeline(features, model):
    numeric_features = [f for f in features if f != "City"]

    preprocessor = ColumnTransformer([
        ("city", OneHotEncoder(handle_unknown="ignore"), ["City"]),
        ("numeric", SimpleImputer(strategy="median"), numeric_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("tree", model)
    ])


def get_feature_names(model):
    return model.named_steps["preprocessor"].get_feature_names_out()


def save_regression_plots(model, X_test, y_test, pred, name, tag):
    tree = model.named_steps["tree"]
    feature_names = get_feature_names(model)

    # Feature importance.
    importance = pd.Series(
        tree.feature_importances_,
        index=feature_names
    ).sort_values()

    plt.figure(figsize=(9, 6))
    importance.tail(15).plot.barh(color="#2e9e4f")
    plt.title(f"Feature Importance - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}_importance.png", dpi=150)
    plt.close()

    # Actual versus predicted AQI.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, s=8, alpha=0.5)
    limit = [
        min(float(y_test.min()), float(np.min(pred))),
        max(float(y_test.max()), float(np.max(pred)))
    ]
    plt.plot(limit, limit, "r--")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(f"Actual vs Predicted - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}_actual_vs_pred.png", dpi=150)
    plt.close()

    # Decision tree visualisation.
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        max_depth=3,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=7
    )
    plt.title(f"Decision Tree - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}_tree.png", dpi=150)
    plt.close()


def run_regression(train, test, features, target, name, tag):
    train = train.dropna(subset=[target]).copy()
    test = test.dropna(subset=[target]).copy()

    if len(train) < 20 or len(test) < 2:
        print(f"\nSkipping {name}: insufficient training or testing data.")
        return None

    train = train.sort_values(["Date", "City"]).reset_index(drop=True)
    test = test.sort_values(["Date", "City"]).reset_index(drop=True)

    model = make_pipeline(
        features,
        DecisionTreeRegressor(random_state=RANDOM_STATE)
    )

    grid = GridSearchCV(
        model,
        {
            "tree__max_depth": [3, 4, 6, 8, 10],
            "tree__min_samples_leaf": [1, 5, 10, 20]
        },
        cv=make_date_cv(train),
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid.fit(train[features], train[target])

    best_model = grid.best_estimator_
    pred = best_model.predict(test[features])

    mae = mean_absolute_error(test[target], pred)
    rmse = np.sqrt(mean_squared_error(test[target], pred))
    r2 = r2_score(test[target], pred)

    print(f"\n[{name}] Decision Tree Regressor")
    print(f"Training rows: {len(train)} | Testing rows: {len(test)}")
    print(f"Best parameters: {grid.best_params_}")
    print(f"MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f}")

    joblib.dump(best_model, f"{OUT}/{tag}_model.pkl")

    save_regression_plots(
        best_model, test[features], test[target],
        pred, name, tag
    )

    # City-wise performance.
    city_results = []

    for city in test["City"].unique():
        mask = test["City"].to_numpy() == city
        actual = test.loc[mask, target]
        predicted = pred[mask]

        city_results.append({
            "City": city,
            "Task": name,
            "Observations": len(actual),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted) if len(actual) > 1 else np.nan
        })

    return {
        "Task": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CityResults": city_results
    }


def run_classification(train, test, features):
    train = train.dropna(subset=["AQI_Bucket"]).copy()
    test = test.dropna(subset=["AQI_Bucket"]).copy()

    if len(train) < 20 or len(test) < 2:
        print("\nSkipping classification: insufficient data.")
        return None

    model = make_pipeline(
        features,
        DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )

    model.fit(train[features], train["AQI_Bucket"])
    pred = model.predict(test[features])
    actual = test["AQI_Bucket"]

    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average="weighted", zero_division=0)
    recall = recall_score(actual, pred, average="weighted", zero_division=0)
    f1 = f1_score(actual, pred, average="weighted", zero_division=0)

    print("\n[Combined AQI Category] Decision Tree Classifier")
    print(f"Training rows: {len(train)} | Testing rows: {len(test)}")
    print(f"Accuracy={accuracy:.3f}")
    print(f"Precision={precision:.3f}")
    print(f"Recall={recall:.3f}")
    print(f"F1={f1:.3f}")

    joblib.dump(model, f"{OUT}/combined_classifier.pkl")

    # Confusion matrix.
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        actual,
        pred,
        ax=ax,
        xticks_rotation=45,
        cmap="Greens"
    )
    plt.tight_layout()
    plt.savefig(f"{OUT}/combined_confusion_matrix.png", dpi=150)
    plt.close()

    return {
        "Task": "AQI Classification",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        default="city_day.csv",
        help="Path to combined city dataset"
    )
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)

    print("Loading combined city dataset...")
    df = load_data(args.csv)

    print(f"\nTotal observations: {len(df)}")
    print(f"Number of cities: {df['City'].nunique()}")
    print(f"Cities: {sorted(df['City'].unique())}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    train, test, cutoff = chronological_split(df)

    time_features = [
        "Year", "Month", "Weekday", "Season",
        "Is_Winter", "Is_Festival_Season"
    ]

    pollutant_features = [p for p in POLLUTANTS if p in df.columns]

    # Same-day AQI: current-day pollutant and date information.
    same_day_features = ["City"] + time_features + pollutant_features

    # Next-day AQI: current-day AQI is also included as an input.
    next_day_features = same_day_features + ["AQI"]

    results = []
    city_results = []

    # Same-day regression.
    result = run_regression(
        train, test, same_day_features,
        "AQI", "Combined Same-Day AQI", "combined_sameday"
    )

    if result:
        city_results.extend(result.pop("CityResults"))
        results.append(result)

    # Classification.
    result = run_classification(train, test, same_day_features)

    if result:
        results.append(result)

    # For next-day prediction, training targets must also fall
    # before the testing cutoff date.
    next_train = train[
        (train["AQI_Next_Day"].notna()) &
        (train["Next_Date"] < cutoff)
    ].copy()

    next_test = test[
        test["AQI_Next_Day"].notna()
    ].copy()

    result = run_regression(
        next_train, next_test, next_day_features,
        "AQI_Next_Day", "Combined Next-Day AQI", "combined_nextday"
    )

    if result:
        city_results.extend(result.pop("CityResults"))
        results.append(result)

    # Save overall performance.
    if results:
        pd.DataFrame(results).to_csv(
            f"{OUT}/combined_model_summary.csv",
            index=False
        )

    # Save city-wise regression performance.
    if city_results:
        pd.DataFrame(city_results).to_csv(
            f"{OUT}/city_wise_performance.csv",
            index=False
        )

    print("\nTraining and evaluation completed.")
    print(f"All results saved in ./{OUT}/")


if __name__ == "__main__":
    main()
