# Airline Passenger Satisfaction Prediction

## Overview
This project aims to predict airline passenger satisfaction (classified as either 'satisfied' or 'neutral/dissatisfied') using the [Airline Passenger Satisfaction dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) from Kaggle. 

The primary objective of this repository is to build and evaluate predictive models entirely in Python, with the ultimate success criteria being the implementation of an autoresearch agent designed specifically to outperform manually tuned baseline machine learning models.

## Project Structure
Below is a description of the core directories and configuration files in this repository:

- `data/raw/`: Raw Kaggle source files kept as the original download reference.
- `data/`: The working train/test data used by this project after resplitting for reproducibility and consistent experiments.
- `scripts/`: Supporting notebook-style workflow for the data split and baseline modeling work that established the initial benchmark.
- `model.py`: The current editable autoresearch model definition used for customer satisfaction prediction.
- `prepare.py`: Shared experiment utilities for loading data, creating the validation split, evaluating models, logging results, and plotting performance.
- `run.py`: Runs a single labeled autoresearch experiment against the current `model.py`.
- `demo.py`: Runs the multi-iteration autoresearch demonstration loop and writes the latest run to `results.tsv`.
- `program.md`: The high-level instructions and constraints for the autoresearch workflow.
- `results.tsv`: The latest run-level experiment log used to generate the current performance plot.
- `performance.png`: The latest visualization generated from `results.tsv`.

## Experiment Log

Track the performance of manual baseline models against the automated research runs below. We use `ROC AUC` as the primary model-selection metric throughout autoresearch; `accuracy` is included as a secondary metric for easier side-by-side viewing.

| Model Type | Creator (User / Autoresearch) | Autoresearch Run # | Runtime (s) | ROC AUC | Accuracy | Preserved/Deleted | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | User | | 3.2949 seconds |0.9197 | NA* | Preserved | Baseline with minimal preprocessing. Max iterations set to 1000. |
| Logistic Regression | Autoresearch |2| 0.24 seconds |0.9295 | 0.8748 | Preserved | 8-iteration run exp-001 baseline from `demo.py`. |
| Logistic Regression Balanced | Autoresearch |2| 0.24 seconds |0.9294 | 0.8691 | Deleted | 8-iteration run exp-002 discarded. `class_weight='balanced'` slightly reduced ROC AUC. |
| Logistic Regression C=0.5 | Autoresearch |2|0.23 seconds |0.9295 | 0.8749 | Deleted | 8-iteration run exp-003 discarded. Essentially tied baseline on ROC AUC. |
| Random Forest | Autoresearch |2|15.27 seconds |0.9920 | 0.9539 | Preserved | 8-iteration run exp-004 kept. `n_estimators=300`, `max_depth=12`, `min_samples_leaf=2`, `n_jobs=1`. |
| Random Forest Full Depth | Autoresearch |2|30.38 seconds |0.9942 | 0.9636 | Preserved | 8-iteration run exp-005 kept and became best overall. `n_estimators=500`, full depth, single-threaded for sandbox safety. |
| Extra Trees | Autoresearch |2|15.57 seconds |0.9924 | 0.9563 | Deleted | 8-iteration run exp-006 discarded. Strong result but below best random forest. |
| Gradient Boosting | Autoresearch |2|27.41 seconds |0.9909 | 0.9523 | Deleted | 8-iteration run exp-007 discarded. Lower ROC AUC than the current best. |
| Extra Trees Full Depth | Autoresearch |2|27.25 seconds |0.9934 | 0.9632 | Deleted | 8-iteration run exp-008 discarded. Close, but still below exp-005. |
| Random Forest Full Depth | Autoresearch |3|28.58 seconds |0.9942 | 0.9636 | Preserved | 4-iteration run baseline using the previously preserved best `RandomForestClassifier(n_estimators=500, n_jobs=1)`. |
| Random Forest Full Depth + Leaf 2 | Autoresearch |3|26.45 seconds |0.9940 | 0.9634 | Deleted | Iteration 2 discarded. Adding `min_samples_leaf=2` slightly reduced ROC AUC relative to the run baseline. |
| Random Forest Balanced Subsample | Autoresearch |3|32.29 seconds |0.9941 | 0.9632 | Deleted | Iteration 3 discarded. `class_weight='balanced_subsample'` did not beat the baseline ROC AUC. |
| Random Forest Log Loss | Autoresearch |3|28.63 seconds |0.9944 | 0.9640 | Preserved | Iteration 4 kept and became the new best model. Switched the forest criterion to `log_loss`. |
| Random Forest Log Loss | Autoresearch |4|26.97 seconds |0.9944 | 0.9640 | Preserved | 3-iteration run baseline using the current preserved best `RandomForestClassifier(criterion='log_loss', n_estimators=500, n_jobs=1)`. |
| Random Forest + Engineered Aggregate Features | Autoresearch |4|29.48 seconds |0.9943 | 0.9637 | Deleted | Iteration 2 discarded. Added aggregate service and delay features in preprocessing, but ROC AUC stayed below the run baseline. |
| Random Forest + Aggregate Features and Flags | Autoresearch |4|28.19 seconds |0.9943 | 0.9636 | Deleted | Iteration 3 discarded. Added delay and traveler-profile flags on top of engineered aggregates, but still did not beat the baseline ROC AUC. |

\* Accuracy values for the first three logged rows were not preserved because this column was added after those earlier runs had already been documented.

## Latest Performance Plot

![Latest autoresearch performance](performance.png)
