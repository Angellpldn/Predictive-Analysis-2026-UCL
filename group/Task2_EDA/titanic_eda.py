#Copilot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the Titanic dataset
# Use local file from Task1_ingestion data directory
local_path = '../Task1_ingestion/../data/Titanic-Dataset.csv'
abs_path = Path('/Users/angellp/Predictive-Analysis-2026-UCL/group/data/Titanic-Dataset.csv')

if abs_path.exists():
    df = pd.read_csv(abs_path)
    print(f"✓ Loaded Titanic dataset from: {abs_path}")
elif Path(local_path).exists():
    df = pd.read_csv(local_path)
    print(f"✓ Loaded Titanic dataset from: {local_path}")
else:
    print(f"✗ Error: Dataset not found at:")
    print(f"  {abs_path}")
    print(f"  {local_path}")
    exit(1)

# Create output directory for plots
output_dir = Path('plots')
output_dir.mkdir(exist_ok=True)

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# ============================================================================
# PLOT 1: Survival Rate Distribution (Bar Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
survival_counts = df['Survived'].value_counts().sort_index()
survival_labels = ['Did Not Survive', 'Survived']
colors = ['#d62728', '#2ca02c']

bars = ax.bar(survival_labels, survival_counts.values, color=colors, edgecolor='black', linewidth=1.5)

# Add count labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_xlabel('Survival Status', fontsize=12, fontweight='bold')
ax.set_title('Titanic Survival Rate Distribution', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'plot1_survival_dist.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plot1_survival_dist.png")
plt.close()

# ============================================================================
# PLOT 2: Survival Rate by Pclass and Sex (Grouped Bar Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Create pivot table for survival by Pclass and Sex
survival_by_class_sex = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()

# Plot grouped bar chart
x = np.arange(len(survival_by_class_sex.index))
width = 0.35

bars1 = ax.bar(x - width/2, survival_by_class_sex['female'], width, 
               label='Female', color='#ff9999', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, survival_by_class_sex['male'], width,
               label='Male', color='#66b3ff', edgecolor='black', linewidth=1.5)

# Add percentage labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Survival Rate', fontsize=12, fontweight='bold')
ax.set_xlabel('Passenger Class', fontsize=12, fontweight='bold')
ax.set_title('Survival Rate by Passenger Class and Sex', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(['Class 1', 'Class 2', 'Class 3'])
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'plot2_survival_by_class_sex.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot2_survival_by_class_sex.png")
plt.close()

# ============================================================================
# PLOT 3: Age Distribution with Survival Overlay (Histogram)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histograms for each survival group
age_survived = df[df['Survived'] == 1]['Age'].dropna()
age_not_survived = df[df['Survived'] == 0]['Age'].dropna()

ax.hist(age_not_survived, bins=30, alpha=0.6, label='Did Not Survive', 
        color='#d62728', edgecolor='black', linewidth=1.2)
ax.hist(age_survived, bins=30, alpha=0.6, label='Survived',
        color='#2ca02c', edgecolor='black', linewidth=1.2)

ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Age Distribution with Survival Overlay', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'plot3_age_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot3_age_distribution.png")
plt.close()

# ============================================================================
# PLOT 4: Missing Value Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate missing value percentages
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percent': (df.isnull().sum() / len(df)) * 100
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)

if len(missing_data) > 0:
    # Create a heatmap representation
    missing_matrix = pd.DataFrame({col: [df[col].isnull().sum() / len(df) * 100] for col in missing_data['Column']})
    
    sns.heatmap(missing_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Missing Percentage (%)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_ylabel('', fontsize=12, fontweight='bold')
    ax.set_title('Missing Value Heatmap', fontsize=14, fontweight='bold', pad=20)
else:
    ax.text(0.5, 0.5, 'No Missing Values Found', 
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Missing Value Heatmap', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'plot4_missing_values.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot4_missing_values.png")
plt.close()

# ============================================================================
# WRITTEN SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS SUMMARY - TITANIC SURVIVAL PREDICTION")
print("="*80)

# 1. Class Imbalance in Target Variable
print("\n1. CLASS IMBALANCE IN TARGET VARIABLE")
print("-" * 80)
survived_count = df['Survived'].sum()
not_survived_count = len(df) - survived_count
total = len(df)
print(f"Total Passengers: {total}")
print(f"Survived: {survived_count} ({survived_count/total*100:.2f}%)")
print(f"Did Not Survive: {not_survived_count} ({not_survived_count/total*100:.2f}%)")
print(f"Class Imbalance Ratio: 1:{not_survived_count/survived_count:.2f}")
print("\nInsight: The dataset is moderately imbalanced with approximately 38% survival rate.")
print("This should be considered when selecting evaluation metrics and model parameters.")

# 2. Key Predictors of Survival
print("\n2. KEY PREDICTORS OF SURVIVAL BASED ON PLOTS")
print("-" * 80)

# Survival by Sex
print("\nSex Impact:")
sex_survival = df.groupby('Sex')['Survived'].agg(['mean', 'count'])
print(sex_survival)
print(f"→ Females had {sex_survival.loc['female', 'mean']*100:.1f}% survival rate")
print(f"→ Males had {sex_survival.loc['male', 'mean']*100:.1f}% survival rate")
print("→ SEX is a strong predictor - 'Women and children first' evacuation policy evident")

# Survival by Pclass
print("\nPassenger Class Impact:")
pclass_survival = df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
print(pclass_survival)
print(f"→ 1st Class: {pclass_survival.loc[1, 'mean']*100:.1f}% survival")
print(f"→ 2nd Class: {pclass_survival.loc[2, 'mean']*100:.1f}% survival")
print(f"→ 3rd Class: {pclass_survival.loc[3, 'mean']*100:.1f}% survival")
print("→ PCLASS is a strong predictor - higher classes had better survival rates")

# Survival by Age
print("\nAge Impact:")
age_stats = df[df['Age'].notna()].groupby(df['Survived'])['Age'].describe()
print(f"→ Survivors mean age: {df[df['Survived']==1]['Age'].mean():.1f} years")
print(f"→ Non-survivors mean age: {df[df['Survived']==0]['Age'].mean():.1f} years")
print("→ AGE shows some predictive power - younger passengers had better survival chances")

# 3. Missing Value Concerns and Leakage Risks
print("\n3. MISSING VALUE CONCERNS AND LEAKAGE RISKS")
print("-" * 80)
print("\nMissing Values Summary:")
for idx, row in missing_data.iterrows():
    print(f"→ {row['Column']}: {row['Missing_Count']} missing ({row['Missing_Percent']:.2f}%)")

print("\nMissing Value Handling Recommendations:")
print("→ Age: 19.87% missing - consider imputation (median, KNN, or modeling-based)")
print("→ Cabin: 77.1% missing - too sparse; consider deriving feature (Deck) or dropping")
print("→ Embarked: 0.22% missing - few values; safe to drop or use mode imputation")

print("\nLeakage Risks:")
print("→ None identified in standard Titanic dataset")
print("→ All features are passenger characteristics known before boarding")
print("→ Cabin feature should be handled carefully (77% missing) to avoid bias")

# 4. Outliers Identified
print("\n4. OUTLIERS IDENTIFIED")
print("-" * 80)

# Age outliers
age_Q1 = df['Age'].quantile(0.25)
age_Q3 = df['Age'].quantile(0.75)
age_IQR = age_Q3 - age_Q1
age_outliers = df[(df['Age'] < age_Q1 - 1.5*age_IQR) | (df['Age'] > age_Q3 + 1.5*age_IQR)]
print(f"\nAge Outliers (IQR method):")
print(f"→ Lower bound: {age_Q1 - 1.5*age_IQR:.1f} years")
print(f"→ Upper bound: {age_Q3 + 1.5*age_IQR:.1f} years")
print(f"→ Number of outliers: {len(age_outliers)} ({len(age_outliers)/len(df[df['Age'].notna()])*100:.2f}%)")
print(f"→ Max age: {df['Age'].max():.0f} years (potential data quality issue if >120)")

# Fare outliers
fare_Q1 = df['Fare'].quantile(0.25)
fare_Q3 = df['Fare'].quantile(0.75)
fare_IQR = fare_Q3 - fare_Q1
fare_outliers = df[(df['Fare'] < fare_Q1 - 1.5*fare_IQR) | (df['Fare'] > fare_Q3 + 1.5*fare_IQR)]
print(f"\nFare Outliers (IQR method):")
print(f"→ Lower bound: {fare_Q1 - 1.5*fare_IQR:.2f}")
print(f"→ Upper bound: {fare_Q3 + 1.5*fare_IQR:.2f}")
print(f"→ Number of outliers: {len(fare_outliers)} ({len(fare_outliers)/len(df)*100:.2f}%)")
print(f"→ Max fare: {df['Fare'].max():.2f} (premium 1st class tickets)")

print("\nOutlier Handling Recommendations:")
print("→ Age outliers appear reasonable (legitimate elderly passengers)")
print("→ Fare outliers legitimate (luxury tickets for wealthy passengers)")
print("→ Recommend keeping outliers as they represent real business characteristics")

print("\n" + "="*80)
print("Analysis Complete! Plots saved in 'plots' directory.")
print("="*80)
