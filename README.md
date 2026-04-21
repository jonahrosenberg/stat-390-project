# Airline Passenger Satisfaction Prediction

## Overview
This project aims to predict airline passenger satisfaction (classified as either 'satisfied' or 'neutral/dissatisfied') using the [Airline Passenger Satisfaction dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) from Kaggle. 

The primary objective of this repository is to build and evaluate predictive models entirely in Python, with the ultimate success criteria being the implementation of an autoresearch agent designed specifically to outperform manually tuned baseline machine learning models.

## Project Structure
Below is a description of the core directories and configuration files in this repository:

* **`autoresearch/`**: Contains the automated research logs, agent configurations, and outputs for the autoresearch system aimed at beating the baseline.
* **`data/`**: Stores the raw, interim, and processed dataset files downloaded from Kaggle. *Note: Data files are excluded from version control via `.gitignore`.*
* **`manual-models/`**: Includes manually crafted baseline machine learning models and exploratory scripts used as the benchmark for the autoresearch agent.
* **`memos/`**: Contains project documentation, analytical notes, and progress reports regarding data insights and model performance.
* **`model-fits/`**: Stores serialized versions of trained models, evaluation metrics, and performance plots.
* **`scripts/`**: The primary source code directory containing Python scripts for data preprocessing, feature engineering, model training, and evaluation.
* **`.gitignore`**: Specifies untracked files and directories to ignore in Git version control.
* **`README.md`**: The primary documentation file for the project (this document).

## Tech Stack
This project is developed entirely in **Python**. It relies on standard data science libraries for preprocessing, building the manual baseline models, and constructing the automated research workflow.