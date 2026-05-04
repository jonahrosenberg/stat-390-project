"""
Demo: simulate an AutoResearch agent loop for customer satisfaction with 8 iterations.

This script demonstrates the full keep/discard workflow:
  1. Run a baseline classifier
  2. Try modifications one by one
  3. Keep improvements, discard regressions
  4. Plot the trajectory

Usage: python demo.py
"""
import os
import time

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare import (
    RESULTS_FILE,
    evaluate,
    get_next_autoresearch_run,
    load_data,
    log_result,
    plot_results,
)


def make_preprocessor():
    """Build a reusable preprocessing pipeline for mixed-type tabular data."""
    numeric_features = [
        "Age",
        "Flight Distance",
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Inflight service",
        "Cleanliness",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
    ]
    categorical_features = [
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


ITERATIONS = [
    {
        "id": 1,
        "description": "baseline: LogisticRegression",
        "model": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    },
    {
        "id": 2,
        "description": "LogisticRegression(class_weight='balanced')",
        "model": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
    },
    {
        "id": 3,
        "description": "LogisticRegression(C=0.5)",
        "model": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
        ]),
    },
    {
        "id": 4,
        "description": "RandomForest(n=300, depth=12, leaf=2)",
        "model": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            )),
        ]),
    },
    {
        "id": 5,
        "description": "RandomForest(n=500, full depth) -- risky",
        "model": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=1,
            )),
        ]),
    },
    {
        "id": 6,
        "description": "ExtraTrees(n=400, depth=18)",
        "model": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", ExtraTreesClassifier(
                n_estimators=400,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            )),
        ]),
    },
    {
        "id": 7,
        "description": "GradientBoosting(n=250, depth=3)",
        "model": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.08,
                max_depth=3,
                random_state=42,
            )),
        ]),
    },
    {
        "id": 8,
        "description": "ExtraTrees(n=600, full depth)",
        "model": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", ExtraTreesClassifier(
                n_estimators=600,
                random_state=42,
                n_jobs=1,
            )),
        ]),
    },
]


def main():
    autoresearch_run = get_next_autoresearch_run()
    print(f"Appending to {RESULTS_FILE} as autoresearch run #{autoresearch_run}\n")

    X_train, y_train, X_val, y_val, feature_names = load_data()
    print("Dataset: Airline Passenger Satisfaction")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Features: {list(feature_names)}")
    print(f"{'=' * 70}\n")

    best_auc = -float("inf")

    for iteration in ITERATIONS:
        exp_id = iteration["id"]
        desc = iteration["description"]
        model = iteration["model"]

        print(f"-- Experiment {exp_id}: {desc}")

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        val_accuracy, val_roc_auc = evaluate(model, X_val, y_val)

        if exp_id == 1:
            status = "baseline"
            best_auc = val_roc_auc
            decision_msg = "BASELINE established"
        elif val_roc_auc > best_auc:
            status = "keep"
            improvement = (val_roc_auc - best_auc) * 100
            decision_msg = f"KEEP  (improved AUC by {improvement:.2f} points)"
            best_auc = val_roc_auc
        else:
            status = "discard"
            regression = (best_auc - val_roc_auc) * 100
            decision_msg = f"DISCARD (AUC dropped {regression:.2f} points)"

        log_result(
            f"exp-{exp_id:03d}",
            val_accuracy,
            val_roc_auc,
            status,
            desc,
            autoresearch_run=autoresearch_run,
        )

        print(
            f"   Accuracy: {val_accuracy:.6f}  |  ROC AUC: {val_roc_auc:.6f}  |  Time: {train_time:.2f}s"
        )
        print(f"   >>> {decision_msg}")
        print()

    print(f"{'=' * 70}")
    print(f"Best ROC AUC achieved: {best_auc:.6f}")
    print(f"Autoresearch run #:      {autoresearch_run}")
    print(f"Results saved to:      {RESULTS_FILE}")
    print()

    plot_results("performance.png")
    print("\nDone. Open performance.png to see the trajectory.")


if __name__ == "__main__":
    main()
