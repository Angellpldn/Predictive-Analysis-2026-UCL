import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

df = pd.read_csv('/Users/angellp/Downloads/archive/kaggle_london_house_price_data.csv')


leakage_cols = ['rentEstimate_lowerPrice', 'rentEstimate_upperPrice',
                'saleEstimate_lowerPrice', 'saleEstimate_currentPrice',
                'saleEstimate_upperPrice', 'saleEstimate_valueChange.numericChange',
                'saleEstimate_valueChange.percentageChange']

features = ['bedrooms', 'bathrooms', 'floorAreaSqM', 'livingRooms',
            'latitude', 'longitude', 'tenure', 'propertyType',
            'currentEnergyRating', 'outcode']
target = 'rentEstimate_currentPrice'

df_model = df[features + [target]].dropna(subset=[target]).copy()

print(f"Dataset shape after dropping missing target: {df_model.shape}")

# Step 2: EDA

# 1. Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df_model[target].hist(bins=50, ax=axes[0])
axes[0].set_title('Rent Distribution (raw)')
axes[0].set_xlabel('£/month')

np.log1p(df_model[target]).hist(bins=50, ax=axes[1])
axes[1].set_title('Rent Distribution (log-transformed)')
axes[1].set_xlabel('log(£/month)')
plt.tight_layout()
plt.savefig('rent_distribution.png', dpi=150)
plt.show()

# 2. Missing values heatmap
plt.figure(figsize=(10, 4))
missing = df_model.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df_model) * 100).round(1)
print("\nMissing values:")
print(missing_pct[missing_pct > 0])

sns.barplot(x=missing_pct[missing_pct > 0].values,
            y=missing_pct[missing_pct > 0].index)
plt.title('Missing Values (%)')
plt.tight_layout()
plt.savefig('missing_values.png', dpi=150)
plt.show()

# 3. Correlation heatmap (numeric only)
plt.figure(figsize=(8, 6))
numeric_cols = ['bedrooms', 'bathrooms', 'floorAreaSqM', 'livingRooms',
                'latitude', 'longitude', target]
corr = df_model[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation.png', dpi=150)
plt.show()

# 4. Boxplot for outlier detection (raw and log-transformed)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df_model[target].plot(kind='box', ax=axes[0])
axes[0].set_title('Rent Distribution - Boxplot (Raw)')
axes[0].set_ylabel('£/month')

np.log1p(df_model[target]).plot(kind='box', ax=axes[1])
axes[1].set_title('Rent Distribution - Boxplot (Log-Transformed)')
axes[1].set_ylabel('log(£/month)')
plt.tight_layout()
plt.savefig('rent_boxplot_outliers.png', dpi=150)
plt.show()

# 5. Scatter plots for each numeric feature vs log(rent)
numeric_features_list = ['bedrooms', 'bathrooms', 'floorAreaSqM', 'livingRooms',
                         'latitude', 'longitude']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numeric_features_list):
    axes[idx].scatter(df_model[col], np.log1p(df_model[target]), alpha=0.5, s=20)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('log(Rent)')
    axes[idx].set_title(f'{col} vs log(Rent)')

plt.tight_layout()
plt.savefig('numeric_features_vs_rent.png', dpi=150)
plt.show()

# 6. Bar charts for categorical distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# PropertyType distribution
property_counts = df_model['propertyType'].value_counts()
axes[0].bar(range(len(property_counts)), property_counts.values)
axes[0].set_xticks(range(len(property_counts)))
axes[0].set_xticklabels(property_counts.index, rotation=45, ha='right')
axes[0].set_title('Property Type Distribution')
axes[0].set_ylabel('Count')

# Tenure distribution
tenure_counts = df_model['tenure'].value_counts()
axes[1].bar(range(len(tenure_counts)), tenure_counts.values, color='orange')
axes[1].set_xticks(range(len(tenure_counts)))
axes[1].set_xticklabels(tenure_counts.index, rotation=45, ha='right')
axes[1].set_title('Tenure Distribution')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('categorical_distributions.png', dpi=150)
plt.show()

# 7. Data Quality Summary
print("\n" + "="*60)
print("DATA QUALITY SUMMARY")
print("="*60)

# Missing values percentage
total_cells = df_model.shape[0] * df_model.shape[1]
missing_cells = df_model.isnull().sum().sum()
missing_pct_overall = (missing_cells / total_cells) * 100
print(f"\nOverall Missing Data: {missing_pct_overall:.2f}% ({missing_cells}/{total_cells} cells)")
print("\nMissing % by column:")
for col in df_model.columns:
    pct = (df_model[col].isnull().sum() / len(df_model)) * 100
    if pct > 0:
        print(f"  {col}: {pct:.2f}%")

# Duplicates
duplicates = df_model.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates} ({(duplicates/len(df_model)*100):.2f}%)")

# Outlier counts (using IQR method)
print("\nOutlier Detection (IQR method - values beyond Q1-1.5*IQR or Q3+1.5*IQR):")
for col in numeric_features_list:
    Q1 = df_model[col].quantile(0.25)
    Q3 = df_model[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df_model[col] < lower_bound) | (df_model[col] > upper_bound)).sum()
    outlier_pct = (outliers / len(df_model)) * 100
    print(f"  {col}: {outliers} outliers ({outlier_pct:.2f}%)")

# Target variable outliers
Q1_rent = df_model[target].quantile(0.25)
Q3_rent = df_model[target].quantile(0.75)
IQR_rent = Q3_rent - Q1_rent
lower_bound_rent = Q1_rent - 1.5 * IQR_rent
upper_bound_rent = Q3_rent + 1.5 * IQR_rent
outliers_rent = ((df_model[target] < lower_bound_rent) | (df_model[target] > upper_bound_rent)).sum()
print(f"  {target}: {outliers_rent} outliers ({(outliers_rent/len(df_model)*100):.2f}%)")

print("\n" + "-"*60)
print("LEAKAGE RISK EXPLANATION")
print("-"*60)
print("""
The following columns have been EXCLUDED due to leakage risk:
  - rentEstimate_lowerPrice: Lower bound estimate of target variable
  - rentEstimate_upperPrice: Upper bound estimate of target variable
  - saleEstimate_lowerPrice: Price estimate for sales (not our target)
  - saleEstimate_currentPrice: Current sales estimate (not rental)
  - saleEstimate_upperPrice: Upper price estimate for sales
  - saleEstimate_valueChange.numericChange: Derived from sale price
  - saleEstimate_valueChange.percentageChange: Derived from sale price

REASON FOR EXCLUSION:
These columns contain direct derivatives or estimates of the target variable
(rentEstimate_currentPrice) or alternative pricing estimates that would cause:
  1. Target leakage: Model learns from future/derived values
  2. Inflated performance: CV/test scores wouldn't generalize to real predictions
  3. Violation of temporal ordering: Using estimates instead of actual features

SAFE APPROACH:
Using only original property features (bedrooms, tenure, location, etc.)
ensures the model learns true predictive patterns.
""")

# 8. Outlier check
print(f"\nRent outliers (>£20,000/month): {(df_model[target] > 20000).sum()}")
print(f"Rent min: £{df_model[target].min():.0f}, max: £{df_model[target].max():.0f}")

# ── Step 3: Data Preparation ───────────────────────────────

# 先split，再做任何处理！（防止数据泄露）
X = df_model[features]
y = np.log1p(df_model[target])  # log-transform target

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Pipeline（fit只在train上！）
numeric_features = ['bedrooms', 'bathrooms', 'floorAreaSqM', 'livingRooms',
                    'latitude', 'longitude']
categorical_features = ['tenure', 'propertyType', 'currentEnergyRating', 'outcode']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fit on train only
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed   = preprocessor.transform(X_val)
X_test_processed  = preprocessor.transform(X_test)

print("\nPreprocessing complete!")
print(f"Processed train shape: {X_train_processed.shape}")

# Data validation check
assert X_train_processed.shape[0] == len(y_train), "Mismatch in train set!"
assert not np.isnan(X_train_processed).any(), "NaN found after preprocessing!"
print("Validation checks passed!")

# ── Step 4: Build and Compare Models ───────────────────────

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

print("\n" + "-"*60)
print("MODEL TRAINING AND EVALUATION")
print("-"*60)

# Dictionary to store models and results
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=5)
}

results = []

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train on training set
    model.fit(X_train_processed, y_train)
    
    # Make predictions on validation set
    y_val_pred = model.predict(X_val_processed)
    
    # Calculate metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    # Store results
    results.append({
        'Model': model_name,
        'Val RMSE': val_rmse,
        'Val R²': val_r2
    })
    
    print(f"  ✓ Validation RMSE: {val_rmse:.4f}")
    print(f"  ✓ Validation R²:   {val_r2:.4f}")

# Create comparison table
results_df = pd.DataFrame(results)
print("\n" + "-"*60)
print("MODEL COMPARISON TABLE")
print("-"*60)
print(results_df.to_string(index=False))

# Format with better styling for analysis
print("\n" + "-"*60)
print("DETAILED MODEL COMPARISON")
print("-"*60)
print(f"\n{'Model':<20} {'RMSE':<12} {'R² Score':<12}")
print("-" * 44)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<20} {row['Val RMSE']:<12.4f} {row['Val R²']:<12.4f}")

# Find best model by R²
best_model_idx = results_df['Val R²'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_r2 = results_df.loc[best_model_idx, 'Val R²']
best_rmse = results_df.loc[best_model_idx, 'Val RMSE']

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)
print(f"Model: {best_model_name}")
print(f"  RMSE: {best_rmse:.4f}")
print(f"  R²:   {best_r2:.4f}")
print("\nNote: Target is log-transformed. RMSE is in log(£/month) units.")

# ── Step 5: Hyperparameter Tuning with GridSearchCV ──────────────

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

print("\n" + "-"*60)
print("HYPERPARAMETER TUNING (RANDOM FOREST)")
print("-"*60)

# Define parameter grid

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

print(f"\nSearching over {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])} parameter combinations...")

# GridSearchCV with 5-fold cross-validation
from sklearn.utils import resample
X_train_sample, y_train_sample = resample(
    X_train_processed, y_train, 
    n_samples=int(len(X_train_processed) * 0.2), 
    random_state=42
)

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train_sample, y_train_sample)


# Print best parameters and score
print("\n" + "-"*60)
print("GRIDSEARCHCV RESULTS")
print("-"*60)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV R² Score: {grid_search.best_score_:.4f}")

# Get best model
best_rf_model = grid_search.best_estimator_

# Evaluate on training, validation, and test sets
print("\n" + "-"*60)
print("BEST RF MODEL - COMPREHENSIVE EVALUATION")
print("-"*60)

# Predictions on all sets
y_train_pred = best_rf_model.predict(X_train_processed)
y_val_pred = best_rf_model.predict(X_val_processed)
y_test_pred = best_rf_model.predict(X_test_processed)

# Calculate metrics for all sets
datasets = {
    'Train': (y_train, y_train_pred),
    'Validation': (y_val, y_val_pred),
    'Test': (y_test, y_test_pred)
}

eval_results = []
for dataset_name, (y_true, y_pred) in datasets.items():
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    eval_results.append({
        'Dataset': dataset_name,
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae
    })
    
    print(f"\n{dataset_name} Set:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")

# Evaluation table
eval_df = pd.DataFrame(eval_results)
print("\n" + "-"*60)
print("EVALUATION SUMMARY TABLE")
print("-"*60)
print(eval_df.to_string(index=False))

# ── Step 6: Residual Analysis and Feature Importance ───────────────

print("\n" + "-"*60)
print("GENERATING ANALYSIS VISUALIZATIONS")
print("-"*60)

# Get feature names
feature_names = numeric_features + categorical_features

# 1. Residuals Plot (Predicted vs Actual on Test Set)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Predicted vs Actual
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual log(Rent)')
axes[0, 0].set_ylabel('Predicted log(Rent)')
axes[0, 0].set_title('Test Set: Predicted vs Actual')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals vs Actual
residuals_test = y_test - y_test_pred
axes[0, 1].scatter(y_test, residuals_test, alpha=0.5, s=20)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Actual log(Rent)')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Test Set: Residuals vs Actual')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Residuals Histogram
axes[1, 0].hist(residuals_test, bins=50, edgecolor='black')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Test Set: Residuals Distribution')
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Q-Q Plot (Normality Check)
from scipy import stats
stats.probplot(residuals_test, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Residuals Normality Check')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=150)
print("\n✓ Saved: residuals_analysis.png")
plt.show()

# 2. Feature Importance Chart
fig, ax = plt.subplots(figsize=(10, 6))

# Get feature importances
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # Top 15 features

# Plot top features
top_features = [feature_names[i] for i in indices]
top_importances = importances[indices]

ax.barh(range(len(top_features)), top_importances, color='steelblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features)
ax.set_xlabel('Importance')
ax.set_title('Top 15 Feature Importances (Random Forest)')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("✓ Saved: feature_importance.png")
plt.show()

# 3. Failure Mode Analysis: Properties with Largest Prediction Errors
print("\n" + "-"*60)
print("FAILURE MODE ANALYSIS - LARGEST PREDICTION ERRORS")
print("-"*60)

# Calculate absolute errors on test set
abs_errors = np.abs(residuals_test)
largest_error_indices = np.argsort(abs_errors)[::-1][:20]

# Create failure mode dataframe
failure_data = {
    'Index': largest_error_indices,
    'Actual log(Rent)': y_test.values[largest_error_indices],
    'Predicted log(Rent)': y_test_pred[largest_error_indices],
    'Error': residuals_test.values[largest_error_indices],
    'Abs Error': abs_errors.values[largest_error_indices]
}

failure_df = pd.DataFrame(failure_data)

print("\nTop 20 Properties with Largest Prediction Errors:")
print(failure_df[['Actual log(Rent)', 'Predicted log(Rent)', 'Error', 'Abs Error']].to_string(index=False))

print(f"\nError Statistics:")
print(f"  Mean Absolute Error: {abs_errors.mean():.4f}")
print(f"  Median Absolute Error: {np.median(abs_errors):.4f}")
print(f"  Std Dev of Errors: {abs_errors.std():.4f}")
print(f"  Max Error: {abs_errors.max():.4f}")
print(f"  95th Percentile Error: {np.percentile(abs_errors, 95):.4f}")

# 4. Error Distribution by Actual Value
fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(y_test, abs_errors, alpha=0.5, s=20)
ax.set_xlabel('Actual log(Rent)')
ax.set_ylabel('Absolute Prediction Error')
ax.set_title('Prediction Error by Rental Price Level')
ax.grid(True, alpha=0.3)

# Add percentile lines
ax.axhline(y=np.percentile(abs_errors, 95), color='r', linestyle='--', label='95th percentile')
ax.axhline(y=np.median(abs_errors), color='g', linestyle='--', label='Median')
ax.legend()

plt.tight_layout()
plt.savefig('error_by_price.png', dpi=150)
print("✓ Saved: error_by_price.png")
plt.show()

print("\n" + "-"*60)
print("ANALYSIS COMPLETE")
print("-"*60)
print("\nGenerated files:")
print("  1. residuals_analysis.png - Residual diagnostics")
print("  2. feature_importance.png - Top 15 feature importances")
print("  3. error_by_price.png - Error analysis by rental price")
print("\nKey Findings:")
print(f"  • Best model R² on test set: {eval_df[eval_df['Dataset']=='Test']['R²'].values[0]:.4f}")
print(f"  • Model shows {'consistent' if eval_df[eval_df['Dataset']=='Train']['R²'].values[0] - eval_df[eval_df['Dataset']=='Test']['R²'].values[0] < 0.05 else 'potential overfitting'} performance across datasets")