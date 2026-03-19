import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load dataset and basic info
data_path = "/Users/angellp/Predictive-Analysis-2026-UCL/group/Task1_ingestion/Predictive-Analysis-2026-UCL/group/data/Titanic-Dataset.csv"
df = pd.read_csv(data_path)
print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nDtypes:")
print(df.dtypes)

# 2. Missing values: count and percentage per column
missing_count = df.isna().sum()
missing_pct = df.isna().mean() * 100

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct
}).sort_values(by="missing_count", ascending=False)

print("\nMissing values report (count and %):")
print(missing_report)

# 3. Check for duplicate rows
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

# 4. Validate data types for each column (example heuristic checks)
print("\nBasic dtype validation suggestions:")

# Columns that should reasonably be numeric
likely_numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
for col in likely_numeric_cols:
    if col in df.columns:
        print(f"- {col}: dtype={df[col].dtype}, "
              f"n_unique={df[col].nunique()}, "
              f"sample_values={df[col].dropna().unique()[:5]}")

# Columns that should reasonably be categorical
likely_categorical_cols = ["Sex", "Embarked", "Pclass"]
for col in likely_categorical_cols:
    if col in df.columns:
        print(f"- {col}: dtype={df[col].dtype}, "
              f"n_unique={df[col].nunique()}, "
              f"categories={df[col].dropna().unique()}")

# 5. Preprocessing pipeline for binary classification predicting Survived
# Ensure target column is present
target_col = "Survived"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify numeric and categorical columns automatically
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Numeric preprocessing: impute missing values, then scale
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing: impute missing values, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Full pipeline (preprocessing only; you can add a classifier later)
clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)
])

# Train/test split for binary classification task
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

# Fit preprocessing pipeline on training data
clf_pipeline.fit(X_train, y_train)

# Transform both train and test sets
X_train_prepared = clf_pipeline.transform(X_train)
X_test_prepared = clf_pipeline.transform(X_test)

print("\nPreprocessing complete.")
print("Transformed training feature matrix shape:", X_train_prepared.shape)
print("Transformed test feature matrix shape:", X_test_prepared.shape)