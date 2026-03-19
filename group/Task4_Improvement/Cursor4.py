import os
import glob
import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


RANDOM_STATE = 123


def locate_dataset(filename="Titanic-Dataset.csv"):
    candidates = [
        "data/Titanic-Dataset.csv",
        "Titanic-Dataset.csv",
        "group/data/Titanic-Dataset.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    matches = glob.glob(f"**/{filename}", recursive=True)
    if matches:
        return sorted(set(matches), key=len)[0]

    raise FileNotFoundError(f"Could not find {filename} from current directory: {os.getcwd()}")


def find_col(df, target_name):
    for c in df.columns:
        if str(c).strip().lower() == target_name.lower():
            return c
    return None


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.colmap_ = {}

    def fit(self, X, y=None):
        lower_to_actual = {str(c).strip().lower(): c for c in X.columns}
        self.colmap_ = {
            "name": lower_to_actual.get("name"),
            "sibsp": lower_to_actual.get("sibsp"),
            "parch": lower_to_actual.get("parch"),
            "ticket": lower_to_actual.get("ticket"),
            "cabin": lower_to_actual.get("cabin"),
            "passengerid": lower_to_actual.get("passengerid"),
        }
        return self

    @staticmethod
    def _extract_title(name_value):
        if pd.isna(name_value):
            return "Unknown"
        name_value = str(name_value)
        m = re.search(r",\s*([^\.]+)\.", name_value)
        if m:
            title = m.group(1).strip()
            # Optional grouping of rare titles
            if title in {"Mlle", "Ms"}:
                return "Miss"
            if title == "Mme":
                return "Mrs"
            if title in {"Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"}:
                return "Rare"
            return title
        return "Unknown"

    def transform(self, X):
        X = X.copy()

        name_col = self.colmap_.get("name")
        sibsp_col = self.colmap_.get("sibsp")
        parch_col = self.colmap_.get("parch")
        ticket_col = self.colmap_.get("ticket")
        cabin_col = self.colmap_.get("cabin")
        passengerid_col = self.colmap_.get("passengerid")

        # Feature 1: Title from Name
        if name_col is not None and name_col in X.columns:
            X["Title"] = X[name_col].apply(self._extract_title)
        else:
            X["Title"] = "Unknown"

        # Feature 2: FamilySize from SibSp + Parch + 1
        sibsp = pd.to_numeric(X[sibsp_col], errors="coerce") if sibsp_col in X.columns else 0
        parch = pd.to_numeric(X[parch_col], errors="coerce") if parch_col in X.columns else 0
        X["FamilySize"] = sibsp.fillna(0) + parch.fillna(0) + 1

        # Drop irrelevant/high-missing/high-cardinality columns
        drop_cols = [c for c in [ticket_col, cabin_col, name_col, passengerid_col] if c is not None and c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        return X


def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features)
        ]
    )
    return preprocessor


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    return {"model": name, "accuracy": acc, "f1": f1, "estimator": model}


def main():
    # -------------------------
    # Load data
    # -------------------------
    data_path = locate_dataset()
    df = pd.read_csv(data_path)

    target_col = find_col(df, "Survived")
    if target_col is None:
        raise KeyError(f"'Survived' column not found. Columns: {list(df.columns)}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])

    valid_idx = y.notna()
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].astype(int)

    # 80/20 split as required
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # -------------------------
    # Baseline (Task 3 style)
    # -------------------------
    baseline_preprocessor = build_preprocessor(X_train_raw)

    baseline_pipeline = Pipeline(steps=[
        ("preprocess", baseline_preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    baseline_result = evaluate_model(
        "Baseline LogisticRegression",
        baseline_pipeline,
        X_train_raw, y_train, X_test_raw, y_test
    )

    # -------------------------
    # Improvement 1: Feature Engineering
    # -------------------------
    # Fit-transform train only; transform test with fitted transformer
    fe = TitanicFeatureEngineer()
    X_train_fe = fe.fit_transform(X_train_raw, y_train)
    X_test_fe = fe.transform(X_test_raw)

    fe_preprocessor = build_preprocessor(X_train_fe)

    # -------------------------
    # Improvement 2: Alternative models + CV tuning on train only
    # -------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rf_pipeline = Pipeline(steps=[
        ("preprocess", fe_preprocessor),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    rf_param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 2]
    }

    rf_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    rf_search.fit(X_train_fe, y_train)
    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test_fe)
    rf_result = {
        "model": "RandomForest (tuned, + feature engineering)",
        "accuracy": accuracy_score(y_test, rf_pred),
        "f1": f1_score(y_test, rf_pred, zero_division=0),
        "best_params": rf_search.best_params_,
        "cv_best_f1": rf_search.best_score_
    }

    gb_pipeline = Pipeline(steps=[
        ("preprocess", fe_preprocessor),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    gb_param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__min_samples_split": [2, 10]
    }

    gb_search = GridSearchCV(
        estimator=gb_pipeline,
        param_grid=gb_param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    gb_search.fit(X_train_fe, y_train)
    gb_best = gb_search.best_estimator_
    gb_pred = gb_best.predict(X_test_fe)
    gb_result = {
        "model": "GradientBoosting (tuned, + feature engineering)",
        "accuracy": accuracy_score(y_test, gb_pred),
        "f1": f1_score(y_test, gb_pred, zero_division=0),
        "best_params": gb_search.best_params_,
        "cv_best_f1": gb_search.best_score_
    }

    # -------------------------
    # Print required comparison + rationale
    # -------------------------
    print("Why each change should improve performance:")
    print("1) Feature engineering:")
    print("   - Title from Name captures social status and demographics linked to survival behavior.")
    print("   - FamilySize captures group-travel/protection effects not explicit in SibSp/Parch alone.")
    print("   - Dropping Ticket/Cabin/PassengerId/Name reduces noise, high-cardinality sparsity, and missing-data burden.")
    print("2) Alternative models:")
    print("   - Random Forest captures non-linear interactions and is robust to mixed feature effects.")
    print("   - Gradient Boosting can model complex decision boundaries with strong predictive power.")
    print("3) Hyperparameter tuning with CV on train only:")
    print("   - Selects model complexity using only training folds, improving generalization without test leakage.\n")

    print("Task 3 baseline vs improved models (test set):")
    print(f"- {baseline_result['model']}: Accuracy={baseline_result['accuracy']:.4f}, F1={baseline_result['f1']:.4f}")
    print(f"- {rf_result['model']}: Accuracy={rf_result['accuracy']:.4f}, F1={rf_result['f1']:.4f}")
    print(f"  CV best F1={rf_result['cv_best_f1']:.4f}, best params={rf_result['best_params']}")
    print(f"- {gb_result['model']}: Accuracy={gb_result['accuracy']:.4f}, F1={gb_result['f1']:.4f}")
    print(f"  CV best F1={gb_result['cv_best_f1']:.4f}, best params={gb_result['best_params']}")

    # Optional compact delta report
    print("\nImprovement over baseline (test set):")
    print(f"- RandomForest delta: Accuracy={rf_result['accuracy'] - baseline_result['accuracy']:+.4f}, "
          f"F1={rf_result['f1'] - baseline_result['f1']:+.4f}")
    print(f"- GradientBoosting delta: Accuracy={gb_result['accuracy'] - baseline_result['accuracy']:+.4f}, "
          f"F1={gb_result['f1'] - baseline_result['f1']:+.4f}")


if __name__ == "__main__":
    main()