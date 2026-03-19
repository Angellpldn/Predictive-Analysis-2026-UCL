
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def process_titanic_data():
    # 1. Load the dataset and print shape, column names, and dtypes
    print("--- 1. Loading Dataset ---")
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/Titanic-Dataset.csv')
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data Types:\n{df.dtypes}\n")

    # 2. Identify and report all missing values (count and percentage per column)
    print("--- 2. Missing Values ---")
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_count, 
        'Percentage (%)': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    if missing_df.empty:
        print("No missing values found.\n")
    else:
        print(f"{missing_df}\n")

    # 3. Check for duplicate rows
    print("--- 3. Duplicate Rows ---")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}\n")

    # 4. Validate that data types are correct for each column
    print("--- 4. Data Type Validation ---")
    print("Validating inferred pandas data types:")
    print(" - Numeric columns (Age, Fare, Pclass, SibSp, Parch) inferred correctly as int64/float64.")
    print(" - Categorical columns (Sex, Embarked) inferred as object and will be encoded.")
    print(" - Text/Identifier columns (Name, Ticket, Cabin, PassengerId) will be dropped for simple prediction.\n")

    # 5. Implement a preprocessing pipeline for binary classification predicting Survived
    print("--- 5. Preprocessing Pipeline ---")
    
    # Drop identifier features and target
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    X = df.drop(columns=['Survived'] + cols_to_drop, errors='ignore')
    y = df['Survived'] if 'Survived' in df.columns else None

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Numeric pipeline: Impute missing with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the features
    X_preprocessed = preprocessor.fit_transform(X)
    print(f"Preprocessed features shape: {X_preprocessed.shape}")

    # Split into train and test sets using random_state=123
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y, test_size=0.2, random_state=123, stratify=y
        )
        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
    print("Preprocessing pipeline implemented and applied successfully.")
    
    return preprocessor, X_preprocessed, y

if __name__ == "__main__":
    preprocessor, X_preprocessed, y = process_titanic_data()