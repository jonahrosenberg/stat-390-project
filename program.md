# AutoResearch Agent Instructions

## Objective

Minimize validation error on the airline customer satisfaction task by
improving classification performance on the `satisfaction` target.

Primary metric: **validation ROC AUC** (higher is better)  
Secondary metric: **validation accuracy**

## Dataset

- Training data: `data/train.csv`
- Holdout test data: `data/test.csv`
- Target column: `satisfaction`
- Positive class: `satisfied`

The train and test files include mixed numeric and categorical features.
Typical preprocessing will involve imputing missing values and encoding
categorical columns before fitting a classifier.

## Rules

1. The agent should primarily modify `model.py`
2. `build_model()` must return an sklearn-compatible classifier or pipeline
3. The model must predict the `satisfaction` label
4. Use only local data from the `data/` folder
5. Prefer reproducible models and set `random_state=42` where applicable
6. Keep training practical for a classroom demo on CPU

## Workflow

```text
1. Read the current model in model.py
2. Identify the current or next Autoresearch Run # before starting the run
3. Do not delete results.tsv between runs; append new experiments on top of the existing file
4. Ensure each new results.tsv row includes the autoresearch_run value so experiments can be grouped later
5. Propose one modeling change
6. Edit model.py
7. Run the experiment script for a labeled trial
8. Compare validation ROC AUC against the current best
9. If improved: keep the change and note the new best result
10. If worse: revert model.py to the previous version
11. Repeat with a new idea
```

## Good Search Directions

- Logistic regression with stronger or weaker regularization
- Class weighting if the target is imbalanced
- Random forest or gradient boosting classifiers
- HistGradientBoosting with tuned depth, learning rate, and iterations
- Preprocessing changes such as different imputers or scaling choices

## What Not to Do

- Do not hard-code answers from the validation or test set
- Do not add external data sources
- Do not change the `build_model()` function signature
- Do not assume the target is already numeric
