# Airline Passenger Satisfaction Prediction

## Overview
This project aims to predict airline passenger satisfaction (classified as either 'satisfied' or 'neutral/dissatisfied') using the [Airline Passenger Satisfaction dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) from Kaggle. 

The primary objective of this repository is to build and evaluate predictive models entirely in Python, with the ultimate success criteria being the implementation of an autoresearch agent designed specifically to outperform manually tuned baseline machine learning models.

## Project Structure
Below is a description of the core directories and configuration files in this repository:

## Experiment Log

Track the performance of manual baseline models against the automated research runs below:

| Model Type | Creator (User / Autoresearch) | Runtime (s) | ROC AUC | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | User |3.2949 seconds |0.9197 | Baseline with minimal preprocessing. Max iterations set to 1000. |
| Logistic Regression | Autoresearch |0.31 seconds |0.9295 | Iteration 1 baseline via `run.py`; used updated classification scaffold on `data/train.csv` with one-hot encoding and median/mode imputation. |
| Random Forest | Autoresearch |13.27 seconds |0.9920 | Iteration 2 kept. Switched `model.py` to `RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2)` and adjusted to single-thread execution because sandboxed multiprocessing was blocked. |
