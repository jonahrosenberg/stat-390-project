"""
Run one experiment: build model, train, evaluate, log result.

Usage:
    python run.py "description"              # logs as status=keep
    python run.py "description" --baseline   # logs as status=baseline
    python run.py "description" --discard    # logs as status=discard
"""
import sys
import time
import subprocess
from prepare import get_next_autoresearch_run, load_data, evaluate, log_result


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def main():
    args = sys.argv[1:]
    status = "keep"
    autoresearch_run = None
    description_parts = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        elif a == "--run" and i + 1 < len(args):
            autoresearch_run = args[i + 1]
            i += 1
        else:
            description_parts.append(a)
        i += 1
    description = " ".join(description_parts) if description_parts else "experiment"
    if autoresearch_run is None:
        autoresearch_run = get_next_autoresearch_run()

    X_train, y_train, X_val, y_val, feature_names = load_data()
    print(
        f"Data: {X_train.shape[0]} train, {X_val.shape[0]} val, {len(feature_names)} features"
    )

    from model import build_model

    model = build_model()
    print(f"Model: {model}")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    val_accuracy, val_roc_auc = evaluate(model, X_val, y_val)
    print(f"val_accuracy: {val_accuracy:.6f}")
    print(f"val_roc_auc:  {val_roc_auc:.6f}")

    commit = get_git_hash()
    log_result(
        commit,
        val_accuracy,
        val_roc_auc,
        status,
        description,
        autoresearch_run=autoresearch_run,
    )
    print(f"Result logged to results.tsv (status={status})")


if __name__ == "__main__":
    main()
