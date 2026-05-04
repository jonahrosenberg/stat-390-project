"""
Project setup utilities for airline customer satisfaction classification.

This module handles data loading, train/validation splitting, evaluation,
result logging, and plotting for the AutoResearch loop.
"""
from pathlib import Path
import csv
import os

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
VAL_FRACTION = 0.2
RESULTS_FILE = "results.tsv"
TARGET_COLUMN = "satisfaction"
POSITIVE_LABEL = "satisfied"
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
DROP_COLUMNS = ["Unnamed: 0", "id"]


def get_next_autoresearch_run():
    """Return the next autoresearch run number based on results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        return 1

    max_run = 0
    with open(RESULTS_FILE, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            run_value = (row.get("autoresearch_run") or "").strip()
            if run_value.isdigit():
                max_run = max(max_run, int(run_value))
    return max_run + 1 if max_run else 1


def _encode_target(target_series):
    """Map satisfaction labels to binary values."""
    normalized = target_series.astype(str).str.strip().str.lower()
    return normalized.eq(POSITIVE_LABEL).astype(int)


def _read_dataset(csv_path):
    """Read a dataset and separate features from the binary target."""
    frame = pd.read_csv(csv_path)
    frame = frame.drop(columns=[col for col in DROP_COLUMNS if col in frame.columns])

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {csv_path}.")

    y = _encode_target(frame[TARGET_COLUMN])
    X = frame.drop(columns=[TARGET_COLUMN])
    return X, y


def load_data():
    """
    Load the training CSV and create a stratified train/validation split.

    Returns:
        X_train, y_train, X_val, y_val, feature_names
    """
    X, y = _read_dataset(TRAIN_PATH)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return X_train, y_train, X_val, y_val, X.columns.tolist()


def load_test_data():
    """Load the held-out test CSV for later final evaluation."""
    return _read_dataset(TEST_PATH)


def evaluate(model, X_val, y_val):
    """Compute validation accuracy and ROC AUC."""
    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_val)
    else:
        y_score = y_pred

    accuracy = float(accuracy_score(y_val, y_pred))
    roc_auc = float(roc_auc_score(y_val, y_score))
    return accuracy, roc_auc


def log_result(
    experiment_id,
    val_accuracy,
    val_roc_auc,
    status,
    description,
    autoresearch_run="",
):
    """Append one row to results.tsv."""
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if not file_exists:
            writer.writerow(
                [
                    "experiment",
                    "autoresearch_run",
                    "val_accuracy",
                    "val_roc_auc",
                    "status",
                    "description",
                ]
            )
        writer.writerow([
            experiment_id,
            autoresearch_run,
            f"{val_accuracy:.6f}",
            f"{val_roc_auc:.6f}",
            status,
            description,
        ])


def plot_results(save_path="performance.png"):
    """Plot the best validation accuracy and ROC AUC from each autoresearch run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    rows = []
    with open(RESULTS_FILE, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            run_value = (row.get("autoresearch_run") or "").strip()
            if not run_value:
                continue
            rows.append({
                "experiment": row["experiment"],
                "autoresearch_run": run_value,
                "val_accuracy": float(row["val_accuracy"]),
                "val_roc_auc": float(row["val_roc_auc"]),
                "status": row["status"],
                "description": row["description"],
            })

    if not rows:
        print("No autoresearch_run-tagged rows found in results.tsv.")
        return

    frame = pd.DataFrame(rows)
    frame["_run_sort"] = pd.to_numeric(frame["autoresearch_run"], errors="coerce")
    frame = frame.sort_values(
        by=["_run_sort", "val_roc_auc", "val_accuracy"],
        ascending=[True, False, False],
        na_position="last",
    )
    best_per_run = frame.drop_duplicates(subset=["autoresearch_run"], keep="first")
    best_per_run = best_per_run.sort_values(
        by=["_run_sort", "autoresearch_run"],
        na_position="last",
    )

    run_labels = best_per_run["autoresearch_run"].tolist()
    accuracies = best_per_run["val_accuracy"].tolist()
    aucs = best_per_run["val_roc_auc"].tolist()
    statuses = best_per_run["status"].tolist()
    descriptions = best_per_run["description"].tolist()
    positions = list(range(len(run_labels)))

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(status, "#95a5a6") for status in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.scatter(
        positions,
        accuracies,
        c=colors,
        s=80,
        zorder=3,
        edgecolors="white",
        linewidth=0.5,
    )
    ax1.plot(positions, accuracies, "k--", alpha=0.2, zorder=2)

    best_accuracy = []
    running_best_accuracy = -float("inf")
    for score in accuracies:
        running_best_accuracy = max(running_best_accuracy, score)
        best_accuracy.append(running_best_accuracy)
    ax1.plot(
        positions,
        best_accuracy,
        color="#2ecc71",
        linewidth=2.5,
        label="Best so far",
    )
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.set_title(
        "Best Model From Each Autoresearch Run",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    ax2.scatter(
        positions,
        aucs,
        c=colors,
        s=80,
        zorder=3,
        edgecolors="white",
        linewidth=0.5,
    )
    ax2.plot(positions, aucs, "k--", alpha=0.2, zorder=2)

    best_auc = []
    running_best_auc = -float("inf")
    for score in aucs:
        running_best_auc = max(running_best_auc, score)
        best_auc.append(running_best_auc)
    ax2.plot(
        positions,
        best_auc,
        color="#2ecc71",
        linewidth=2.5,
        label="Best so far",
    )
    ax2.set_xlabel("Autoresearch Run #", fontsize=12)
    ax2.set_ylabel("Validation ROC AUC", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    run_tick_labels = [f"Run {run}" for run in run_labels]
    ax2.set_xticks(positions)
    ax2.set_xticklabels(run_tick_labels, rotation=0, ha="center", fontsize=9)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="keep (improved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="discard (regressed)"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Best so far"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_results()
