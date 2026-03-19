import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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


def get_col(df, name):
    for c in df.columns:
        if str(c).strip().lower() == name.lower():
            return c
    return None


# Load data
data_path = locate_dataset()
df = pd.read_csv(data_path)

# Resolve target column
target_col = get_col(df, "Survived")
if target_col is None:
    raise KeyError(f"'Survived' column not found. Available columns: {list(df.columns)}")

# Split features/target
X = df.drop(columns=[target_col])
y = pd.to_numeric(df[target_col], errors="coerce")

# Drop rows with missing target
valid_idx = y.notna()
X = X.loc[valid_idx].copy()
y = y.loc[valid_idx].astype(int)

# Train/test split (80/20, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# Identify feature types from TRAIN only
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Preprocessing pipelines (fit on train only through Pipeline)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Baseline model
model = LogisticRegression(max_iter=1000, random_state=123)

# Full pipeline: preprocessing + model
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Fit only on training data
clf.fit(X_train, y_train)

# Evaluate on test data only
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix:")
print(cm)
