"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for airline customer satisfaction classification.
The function build_model() must return an sklearn-compatible estimator.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])
