import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 0. Load dataset
data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/Titanic-Dataset.csv"
)

df = pd.read_csv(data_path)

# 1. Print shape, column names, and dtypes
print("=== 1. DATASET OVERVIEW ===")
print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# 2. Missing values: count and percentage per column
print("\n=== 2. MISSING VALUES ===")
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent
}).sort_values(by="missing_count", ascending=False)

print(missing_report)

# 3. Check duplicate rows
print("\n=== 3. DUPLICATE ROWS ===")
duplicate_count = df.duplicated().sum()
print("Number of duplicate rows:", duplicate_count)

# 4. Validate data types
print("\n=== 4. DATA TYPE VALIDATION ===")
expected_types = {
    "PassengerId": "integer",
    "Survived": "integer",
    "Pclass": "integer",
    "Name": "string",
    "Sex": "string",
    "Age": "float",
    "SibSp": "integer",
    "Parch": "integer",
    "Ticket": "string",
    "Fare": "float",
    "Cabin": "string",
    "Embarked": "string"
}

validation_results = []

for col in df.columns:
    actual_dtype = str(df[col].dtype)
    expected_type = expected_types.get(col, "unknown")

    if expected_type == "integer":
        valid = pd.api.types.is_integer_dtype(df[col])
    elif expected_type == "float":
        valid = pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col])
    elif expected_type == "string":
        valid = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    else:
        valid = False

    validation_results.append({
        "column": col,
        "expected_type": expected_type,
        "actual_dtype": actual_dtype,
        "valid": valid
    })

dtype_validation = pd.DataFrame(validation_results)
print(dtype_validation)

# 5. Preprocessing pipeline for binary classification
print("\n=== 5. PREPROCESSING PIPELINE ===")

X = df.drop(columns=["Survived"])
y = df["Survived"]

# Drop identifier column
if "PassengerId" in X.columns:
    X = X.drop(columns=["PassengerId"])

categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)
print("X_train shape before preprocessing:", X_train.shape)
print("X_test shape before preprocessing:", X_test.shape)
print("X_train shape after preprocessing:", X_train_processed_df.shape)
print("X_test shape after preprocessing:", X_test_processed_df.shape)

print("\nFirst 5 rows of processed training data:")
print(X_train_processed_df.head())