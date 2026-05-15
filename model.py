"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for airline customer satisfaction classification.
The function build_model() must return an sklearn-compatible estimator.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def build_model():
    """Return an sklearn Pipeline for customer satisfaction prediction."""
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

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("model", XGBClassifier(
            booster="gbtree",
            n_estimators=1100,
            max_depth=0,
            max_leaves=72,
            grow_policy="lossguide",
            learning_rate=0.023,
            subsample=0.89,
            colsample_bytree=0.83,
            colsample_bylevel=0.92,
            colsample_bynode=0.86,
            min_child_weight=1.4,
            gamma=0.025,
            reg_alpha=0.03,
            reg_lambda=1.45,
            max_bin=320,
            max_delta_step=0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )),
    ])
