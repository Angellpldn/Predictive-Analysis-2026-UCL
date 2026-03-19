import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "Titanic-Dataset.csv")

df = pd.read_csv(data_path)

def extract_title(name):
    match = re.search(r",\s*([^\.]+)\.", name)
    return match.group(1).strip() if match else "Unknown"

df["Title"] = df["Name"].astype(str).apply(extract_title)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

num_features = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
cat_features = ["Pclass", "Sex", "Embarked", "Title"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features),
])

# Baseline Logistic Regression
baseline_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=123)),
])

baseline_pipe.fit(X_train, y_train)

y_pred_base = baseline_pipe.predict(X_test)
base_acc = accuracy_score(y_test, y_pred_base)
base_f1 = f1_score(y_test, y_pred_base)

# Random Forest with hyperparameter tuning
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=123)),
])

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
}

rf_cv = GridSearchCV(
    rf_pipe,
    rf_params,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True,
)

rf_cv.fit(X_train, y_train)

y_pred_rf = rf_cv.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

# Gradient Boosting with hyperparameter tuning
gb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=123)),
])

gb_params = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
}

gb_cv = GridSearchCV(
    gb_pipe,
    gb_params,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True,
)

gb_cv.fit(X_train, y_train)

y_pred_gb = gb_cv.predict(X_test)

gb_acc = accuracy_score(y_test, y_pred_gb)

gb_f1 = f1_score(y_test, y_pred_gb)

print("baseline_accuracy", base_acc)
print("baseline_f1", base_f1)
print("rf_accuracy", rf_acc)
print("rf_f1", rf_f1)
print("gb_accuracy", gb_acc)
print("gb_f1", gb_f1)
