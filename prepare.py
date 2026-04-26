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


def log_result(experiment_id, val_accuracy, val_roc_auc, status, description):
    """Append one row to results.tsv."""
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if not file_exists:
            writer.writerow(
                ["experiment", "val_accuracy", "val_roc_auc", "status", "description"]
            )
        writer.writerow([
            experiment_id,
            f"{val_accuracy:.6f}",
            f"{val_roc_auc:.6f}",
            status,
            description,
        ])


def plot_results(save_path="performance.png"):
    """Plot validation accuracy and ROC AUC over experiments from results.tsv."""
    import matplotlib.pyplot as plt

    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    experiments, accuracies, aucs, statuses, descriptions = [], [], [], [], []
    with open(RESULTS_FILE, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            experiments.append(row["experiment"])
            accuracies.append(float(row["val_accuracy"]))
            aucs.append(float(row["val_roc_auc"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(status, "#95a5a6") for status in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.scatter(
        range(len(accuracies)),
        accuracies,
        c=colors,
        s=80,
        zorder=3,
        edgecolors="white",
        linewidth=0.5,
    )
    ax1.plot(range(len(accuracies)), accuracies, "k--", alpha=0.2, zorder=2)

    best_accuracy = []
    running_best_accuracy = -float("inf")
    for score in accuracies:
        running_best_accuracy = max(running_best_accuracy, score)
        best_accuracy.append(running_best_accuracy)
    ax1.plot(
        range(len(accuracies)),
        best_accuracy,
        color="#2ecc71",
        linewidth=2.5,
        label="Best so far",
    )
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.set_title(
        "AutoResearch Demo: Airline Customer Satisfaction Classification",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    ax2.scatter(
        range(len(aucs)),
        aucs,
        c=colors,
        s=80,
        zorder=3,
        edgecolors="white",
        linewidth=0.5,
    )
    ax2.plot(range(len(aucs)), aucs, "k--", alpha=0.2, zorder=2)

    best_auc = []
    running_best_auc = -float("inf")
    for score in aucs:
        running_best_auc = max(running_best_auc, score)
        best_auc.append(running_best_auc)
    ax2.plot(
        range(len(aucs)),
        best_auc,
        color="#2ecc71",
        linewidth=2.5,
        label="Best so far",
    )
    ax2.set_xlabel("Experiment #", fontsize=12)
    ax2.set_ylabel("Validation ROC AUC", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    short_labels = [desc[:22] + ".." if len(desc) > 24 else desc for desc in descriptions]
    ax2.set_xticks(range(len(experiments)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)

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
