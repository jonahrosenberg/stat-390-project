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
7. Automatically stop and discard any model trial that runs for more than 10 minutes (600 seconds)

## Workflow

```text
1. Read the current model in model.py
2. Identify the current or next Autoresearch Run # before starting the run
3. Do not delete results.tsv between runs; append new experiments on top of the existing file
4. Ensure each new results.tsv row includes the autoresearch_run value so experiments can be grouped later
5. Propose one modeling change
6. Edit model.py
7. Run the experiment script for a labeled trial
8. If the model trial runs longer than 600 seconds, stop it, count it as a completed iteration for that run, and revert model.py to the previous version
9. For any timeout trial, add a results.tsv row with status `Failure - Time`; if validation metrics were not produced, record them as `NA`
10. Compare validation ROC AUC against the current best for trials that finish within the time limit
11. If improved: keep the change and note the new best result
12. If worse: revert model.py to the previous version
13. Append the experiment result to the README.md experiment log with the correct run number, metrics, and preserved/deleted/failure outcome
14. For timeout trials, the README.md `Preserved/Deleted` column should read `Failure - Time`
15. Refresh performance.png after the run so it reflects the latest best-per-run history from results.tsv
16. Refresh performance_all_models.png after the run so it reflects all logged experiments from results.tsv
17. Ensure README.md continues to embed the most recent performance.png and performance_all_models.png
18. Repeat with a new idea
```

## Good Search Directions

- Logistic regression with stronger or weaker regularization
- Class weighting if the target is imbalanced
- Random forest or gradient boosting classifiers
- HistGradientBoosting with tuned depth, learning rate, and iterations
- Preprocessing changes such as different imputers or scaling choices
- Added for Run 4: Consider using some preprocessing in model.py to achieve best results. There is a lot of domain knowledge you can use surrounding the variables to make a better prediction.
- Added for Run 6: Consider using cross validation in the model.py for better hyperparameter tuning. This should allow for more hypertuning. I understand that this will add run time as a result, that is okay. Please perform all of this cross validation in model.py and never in any other script.
- Added for Run 7: Consider using a gradient boosting tree for the first three tries of the autoresearch run. Try changing the learn rate and max depth if we are not improving at first.
- Added for Run 9: Lower the cross-validation threshold to a smaller amount so that the models take less time. Instead, prioritize hyperparameter tuning to ensure we're finding optimal values. Search in the areas surrounding the parameters we know work, and search for marginal gains over sweeping improvements that come from large changes.
- Added for Run 10: It has become clear that a random forest model is a superior choice. Let's try to add hyperparameters to the model. There is so much to tune, and we do not just have to accept the default values. Please tune 5-10 parameters per iteration in a space filling grid (or other grid of your choosing).

## What Not to Do

- Do not hard-code answers from the validation or test set
- Do not add external data sources
- Do not change the `build_model()` function signature
- Do not assume the target is already numeric
