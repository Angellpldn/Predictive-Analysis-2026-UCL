import os
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# =========================================================
# Configuration
# =========================================================
RANDOM_STATE = 123
file_path = "/Users/angellp/Predictive-Analysis-2026-UCL/group/data/Titanic-Dataset.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at: {file_path}")

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(file_path)

if "Survived" not in df.columns:
    raise ValueError("Target column 'Survived' not found in dataset.")

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# =========================================================
# Compatibility helper for OneHotEncoder
# =========================================================
def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# =========================================================
# Custom feature engineering
# =========================================================
class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "Name" in X.columns:
            X["Title"] = X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
            X["Title"] = X["Title"].fillna("Unknown").str.strip()
            X["Title"] = X["Title"].replace({
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs"
            })
            rare_titles = [
                "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
                "Rev", "Sir", "Jonkheer", "Dona"
            ]
            X["Title"] = X["Title"].replace(rare_titles, "Rare")

        if "SibSp" in X.columns and "Parch" in X.columns:
            X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        elif "SibSp" in X.columns:
            X["FamilySize"] = X["SibSp"] + 1
        elif "Parch" in X.columns:
            X["FamilySize"] = X["Parch"] + 1
        else:
            X["FamilySize"] = 1

        drop_cols = [col for col in ["PassengerId", "Name", "Ticket", "Cabin"] if col in X.columns]
        X = X.drop(columns=drop_cols)

        return X

# =========================================================
# Preprocessor builder
# =========================================================
def build_preprocessor(X_sample):
    numeric_features = X_sample.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X_sample.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_onehot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor

# =========================================================
# Task 3 Baseline: Logistic Regression without feature engineering
# =========================================================
baseline_X_train = X_train.copy()
baseline_X_test = X_test.copy()

baseline_drop_cols = [col for col in ["PassengerId", "Name", "Ticket", "Cabin"] if col in baseline_X_train.columns]
baseline_X_train = baseline_X_train.drop(columns=baseline_drop_cols)
baseline_X_test = baseline_X_test.drop(columns=baseline_drop_cols)

baseline_preprocessor = build_preprocessor(baseline_X_train)

baseline_model = Pipeline(steps=[
    ("preprocessor", baseline_preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

baseline_model.fit(baseline_X_train, y_train)
baseline_pred = baseline_model.predict(baseline_X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

# =========================================================
# Improved Logistic Regression with feature engineering
# =========================================================
fe = TitanicFeatureEngineer()
X_train_fe_sample = fe.fit_transform(X_train)
improved_preprocessor = build_preprocessor(X_train_fe_sample)

improved_logreg = Pipeline(steps=[
    ("feature_engineering", TitanicFeatureEngineer()),
    ("preprocessor", improved_preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

improved_logreg.fit(X_train, y_train)
improved_logreg_pred = improved_logreg.predict(X_test)

improved_logreg_accuracy = accuracy_score(y_test, improved_logreg_pred)
improved_logreg_f1 = f1_score(y_test, improved_logreg_pred)

# =========================================================
# Random Forest with train-only CV tuning
# =========================================================
rf_pipeline = Pipeline(steps=[
    ("feature_engineering", TitanicFeatureEngineer()),
    ("preprocessor", improved_preprocessor),
    ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
])

rf_param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

rf_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True
)

rf_search.fit(X_train, y_train)
rf_best_model = rf_search.best_estimator_
rf_pred = rf_best_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# =========================================================
# Gradient Boosting with train-only CV tuning
# =========================================================
gb_pipeline = Pipeline(steps=[
    ("feature_engineering", TitanicFeatureEngineer()),
    ("preprocessor", improved_preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE))
])

gb_param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__max_depth": [2, 3],
    "classifier__subsample": [0.8, 1.0]
}

gb_search = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=gb_param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True
)

gb_search.fit(X_train, y_train)
gb_best_model = gb_search.best_estimator_
gb_pred = gb_best_model.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

# =========================================================
# Results comparison
# =========================================================
results = pd.DataFrame([
    {
        "Model": "Task 3 Baseline Logistic Regression",
        "Accuracy": baseline_accuracy,
        "F1 Score": baseline_f1
    },
    {
        "Model": "Improved Logistic Regression + Feature Engineering",
        "Accuracy": improved_logreg_accuracy,
        "F1 Score": improved_logreg_f1
    },
    {
        "Model": "Random Forest + Feature Engineering + CV Tuning",
        "Accuracy": rf_accuracy,
        "F1 Score": rf_f1
    },
    {
        "Model": "Gradient Boosting + Feature Engineering + CV Tuning",
        "Accuracy": gb_accuracy,
        "F1 Score": gb_f1
    }
])

print("Changes and why they should improve performance:")
print("1. Extracting Title from Name adds social-status and demographic information associated with survival patterns.")
print("2. Creating FamilySize adds family-structure information that may capture boarding and evacuation behavior.")
print("3. Dropping Ticket and Cabin removes sparse or high-cardinality fields that can add noise.")
print("4. Random Forest can capture nonlinear effects and feature interactions beyond a linear baseline.")
print("5. Gradient Boosting can improve predictive performance by sequentially correcting earlier errors.")
print("6. Hyperparameter tuning with cross-validation on the training set improves model selection without using test data.")
print()

print("Best Random Forest Parameters:")
print(rf_search.best_params_)
print()

print("Best Gradient Boosting Parameters:")
print(gb_search.best_params_)
print()

print("Model Comparison:")
print(results.to_string(index=False))

# Optional save
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task5_outputs")
os.makedirs(output_dir, exist_ok=True)

results.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

with open(os.path.join(output_dir, "improvement_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Changes and why they should improve performance:\n")
    f.write("1. Extracting Title from Name adds social-status and demographic information associated with survival patterns.\n")
    f.write("2. Creating FamilySize adds family-structure information that may capture boarding and evacuation behavior.\n")
    f.write("3. Dropping Ticket and Cabin removes sparse or high-cardinality fields that can add noise.\n")
    f.write("4. Random Forest can capture nonlinear effects and feature interactions beyond a linear baseline.\n")
    f.write("5. Gradient Boosting can improve predictive performance by sequentially correcting earlier errors.\n")
    f.write("6. Hyperparameter tuning with cross-validation on the training set improves model selection without using test data.\n\n")
    f.write("Best Random Forest Parameters:\n")
    f.write(str(rf_search.best_params_) + "\n\n")
    f.write("Best Gradient Boosting Parameters:\n")
    f.write(str(gb_search.best_params_) + "\n\n")
    f.write("Model Comparison:\n")
    f.write(results.to_string(index=False))