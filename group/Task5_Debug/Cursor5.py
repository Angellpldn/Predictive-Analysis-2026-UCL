import os
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


RANDOM_STATE = 123


def locate_titanic_csv():
    candidates = [
        "titanic.csv",
        "Titanic-Dataset.csv",
        "data/Titanic-Dataset.csv",
        "group/data/Titanic-Dataset.csv",
        "group/Task1_ingestion/Predictive-Analysis-2026-UCL/group/data/Titanic-Dataset.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    matches = glob.glob("**/*[Tt]itanic*.csv", recursive=True)
    if matches:
        return sorted(set(matches), key=len)[0]

    raise FileNotFoundError(
        "Could not find Titanic CSV. Put the file in the project or update locate_titanic_csv()."
    )


def main():
    # ------------------------------------------------------------
    # Four methodological errors in the original pipeline
    # ------------------------------------------------------------
    errors = [
        {
            "error": "Data leakage: StandardScaler was fit on the full dataset before train/test split.",
            "why_problematic": "Test-set information leaks into preprocessing, inflating performance estimates."
        },
        {
            "error": "Missing-value handling used fillna(0) globally before splitting.",
            "why_problematic": "A constant 0 can be invalid for columns like Age/Fare and distorts distributions."
        },
        {
            "error": "train_test_split did not set random_state.",
            "why_problematic": "Results are not reproducible across runs."
        },
        {
            "error": "LogisticRegression did not set random_state.",
            "why_problematic": "Model training can be non-reproducible depending on solver/path."
        },
    ]

    print("Identified methodological errors:")
    for i, e in enumerate(errors, start=1):
        print(f"{i}. {e['error']}")
        print(f"   Why problematic: {e['why_problematic']}")

    # ------------------------------------------------------------
    # Corrected pipeline
    # ------------------------------------------------------------
    csv_path = locate_titanic_csv()
    df = pd.read_csv(csv_path)

    X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]]
    y = df["Survived"]

    # Split first to avoid leakage; reproducible with random_state=123
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # All preprocessing fit on train set only via Pipeline
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\nLoaded dataset from:", csv_path)
    print("Corrected pipeline trained successfully.")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()