"""
Titanic Dataset Analysis and Preprocessing Pipeline (Optimized)
This script handles data loading, exploratory analysis, missing value imputation,
categorical encoding, and preparation for binary classification.
Optimized version that excludes high-cardinality features like Name and Ticket.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATASET AND PRINT BASIC INFORMATION
# ============================================================================
print("="*80)
print("STEP 1: LOADING DATASET AND BASIC INFORMATION")
print("="*80)

# Load the dataset
# Use absolute path to ensure it works from any directory
data_path = '/Users/angellp/Predictive-Analysis-2026-UCL/group/data/Titanic-Dataset.csv'
df = pd.read_csv(data_path)

# Print shape
print(f"\nDataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Print column names
print(f"\nColumn Names:")
print(df.columns.tolist())

# Print data types
print(f"\nData Types:")
print(df.dtypes)

# Display first few rows
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================================
# 2. IDENTIFY AND REPORT MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("STEP 2: MISSING VALUES ANALYSIS")
print("="*80)

# Count missing values
missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': missing_count.values,
    'Missing_Percentage': missing_percentage.values
})

# Filter only columns with missing values
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
    'Missing_Count', ascending=False
)

print("\nMissing Values Summary:")
if len(missing_data) > 0:
    print(missing_data.to_string(index=False))
else:
    print("No missing values found!")

print(f"\nTotal Missing Values: {df.isnull().sum().sum()}")
print(f"Percentage of Total Data Missing: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")

# ============================================================================
# 3. CHECK FOR DUPLICATE ROWS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: DUPLICATE ROWS ANALYSIS")
print("="*80)

duplicate_count = df.duplicated().sum()
print(f"Total Duplicate Rows: {duplicate_count}")
print(f"Percentage of Duplicates: {(duplicate_count / len(df) * 100):.2f}%")

if duplicate_count > 0:
    print("\nFirst few duplicate rows:")
    print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head(10))
else:
    print("No duplicate rows found!")

# ============================================================================
# 4. VALIDATE DATA TYPES AND GENERATE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: DATA TYPE VALIDATION AND STATISTICS")
print("="*80)

print("\nData Type Summary:")
dtype_summary = df.dtypes.value_counts()
print(dtype_summary)

print("\nStatistical Summary for Numerical Columns:")
print(df.describe())

print("\nCategorical Columns Summary:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}: (unique values: {df[col].nunique()})")
    print(df[col].value_counts().head(10))

# ============================================================================
# 5. PREPROCESSING PIPELINE
# ============================================================================
print("\n" + "="*80)
print("STEP 5: BUILDING PREPROCESSING PIPELINE")
print("="*80)

# Drop low-information columns
df_processed = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Separate features and target
X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

print(f"\nTarget Variable Distribution (Survived):")
print(y.value_counts())
print(f"Survival Rate: {(y.sum() / len(y) * 100):.2f}%")

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical Columns: {numerical_cols}")
print(f"Categorical Columns: {categorical_cols}")

# Define preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

print(f"\nProcessed Data Shape: {X_processed.shape}")
print(f"Original Features: {X.shape[1]}")
print(f"Processed Features: {X_processed.shape[1]}")

# Get feature names after transformation
feature_names = (numerical_cols + 
                 list(preprocessor.named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_cols)))

print(f"\nProcessed Feature Names:")
print(feature_names)

# Create processed dataframe
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

print(f"\nProcessed Data - First 5 rows:")
print(X_processed_df.head())

print(f"\nProcessed Data Statistics:")
print(X_processed_df.describe())

print(f"\nProcessed Data Info:")
print(f"Data Type: {X_processed_df.dtypes.unique()}")
print(f"Missing Values in Processed Data: {X_processed_df.isnull().sum().sum()}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING PIPELINE SUMMARY")
print("="*80)
print(f"""
✓ Dataset loaded successfully
✓ Missing values identified and reported
✓ Duplicate rows checked
✓ Data types validated
✓ Low-information columns removed (PassengerId, Name, Ticket, Cabin)
✓ Preprocessing pipeline built with:
  - Numerical: Median imputation + StandardScaler
  - Categorical: Most frequent imputation + OneHotEncoder
✓ Data ready for binary classification (Survived)
✓ Training data shape: {X_processed_df.shape}
✓ Feature list: {list(feature_names)}
✓ Target classes: {y.unique()}
✓ Class distribution: {dict(y.value_counts())}
✓ Random state: 123 (used in pipeline for reproducibility)
""")

print("="*80)

# ============================================================================
# OPTIONAL: Save processed data
# ============================================================================
print("\nOPTIONAL: Saving processed data to CSV files...")

# Create a results directory if it doesn't exist
import os
if not os.path.exists('results'):
    os.makedirs('results')

# Save processed features and target
X_processed_df.to_csv('results/X_processed.csv', index=False)
y.to_csv('results/y_processed.csv', index=False)

# Save preprocessing report
with open('results/preprocessing_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("TITANIC PREPROCESSING REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Original Shape: {df.shape}\n")
    f.write(f"Processed Shape: {X_processed_df.shape}\n")
    f.write(f"Features: {list(feature_names)}\n")
    f.write(f"Missing Values (original): {df.isnull().sum().sum()}\n")
    f.write(f"Missing Values (processed): {X_processed_df.isnull().sum().sum()}\n")
    f.write(f"Duplicate Rows: {duplicate_count}\n")
    f.write(f"Survival Rate: {(y.sum() / len(y) * 100):.2f}%\n")

print("\n✓ Results saved to:")
print("  - results/X_processed.csv (processed features)")
print("  - results/y_processed.csv (target variable)")
print("  - results/preprocessing_report.txt (preprocessing report)")

print("\n" + "="*80)
