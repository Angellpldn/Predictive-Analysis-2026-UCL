import os
import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# Load dataset
# -----------------------------
file_path = "/Users/angellp/Predictive-Analysis-2026-UCL/group/data/Titanic-Dataset.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at: {file_path}")

df = pd.read_csv(file_path)

# -----------------------------
# Custom feature engineering
# -----------------------------
class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "Name" in X.columns:
            X["Title"] = X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
            X["Title"] = X["Title"].fillna("Unknown").str.strip()

            X["Title"] = X["Title"].replace(
                ["Mlle", "Ms", "Mme"], ["Miss", "Miss", "Mrs"]
            )

            rare_titles = [
                "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
                "Rev", "Sir", "Jonkheer", "Dona"
            ]
            X["Title"] = X["Title"].replace(rare_titles, "Rare")

        sibsp = X["SibSp"] if "SibSp" in X.columns else 0
        parch = X["Parch"] if "Parch" in X.columns else 0
        X["FamilySize"] = sibsp + parch + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        drop_cols = [col for col in ["PassengerId", "Name", "Ticket", "Cabin"] if col in X.columns]
        X = X.drop(columns=drop_cols)

        return X

# -----------------------------
# Train/test split
# -----------------------------
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

# -----------------------------
# Build preprocessor
# -----------------------------
feature_engineer = TitanicFeatureEngineer()
X_train_fe = feature_engineer.fit_transform(X_train)

numeric_features = X_train_fe.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train_fe.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# -----------------------------
# Baseline model
# -----------------------------
baseline = Pipeline([
    ("feature_engineering", TitanicFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=123))
])

baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

baseline_acc = accuracy_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

# -----------------------------
# Random Forest with CV
# -----------------------------
rf_pipeline = Pipeline([
    ("feature_engineering", TitanicFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=123))
])

rf_params = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10]
}

rf_cv = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_cv.fit(X_train, y_train)
rf_best = rf_cv.best_estimator_
rf_pred = rf_best.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# -----------------------------
# Gradient Boosting with CV
# -----------------------------
gb_pipeline = Pipeline([
    ("feature_engineering", TitanicFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=123))
])

gb_params = {
    "classifier__n_estimators": [100, 200],
    "classifier__learning_rate": [0.05, 0.1]
}

gb_cv = GridSearchCV(
    gb_pipeline,
    gb_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

gb_cv.fit(X_train, y_train)
gb_best = gb_cv.best_estimator_
gb_pred = gb_best.predict(X_test)

gb_acc = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

# -----------------------------
# Results
# -----------------------------
print("Why improvements help:")
print("Title captures social status patterns linked to survival.")
print("FamilySize and IsAlone encode group structure effects.")
print("Tree models capture nonlinear relationships.")
print("Cross-validation improves parameter selection.\n")

print("Baseline Logistic Regression:")
print("Accuracy:", baseline_acc)
print("F1:", baseline_f1)

print("\nRandom Forest:")
print("Best params:", rf_cv.best_params_)
print("Accuracy:", rf_acc)
print("F1:", rf_f1)

print("\nGradient Boosting:")
print("Best params:", gb_cv.best_params_)
print("Accuracy:", gb_acc)
print("F1:", gb_f1)