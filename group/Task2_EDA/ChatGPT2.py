import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# FIXED: Use your explicit absolute path
# -----------------------------
file_path = "/Users/angellp/Predictive-Analysis-2026-UCL/group/data/Titanic-Dataset.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at: {file_path}")

df = pd.read_csv(file_path)

# -----------------------------
# Setup output directory (relative to script)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "eda_outputs")
os.makedirs(output_dir, exist_ok=True)

sns.set_theme(style="whitegrid")

# -----------------------------
# Plot 1: Survival rate distribution
# -----------------------------
plt.figure(figsize=(8, 6))
survival_counts = df["Survived"].value_counts().sort_index()
ax = sns.barplot(x=survival_counts.index, y=survival_counts.values)
ax.set_title("Survival Rate Distribution")
ax.set_xlabel("Survived (0 = No, 1 = Yes)")
ax.set_ylabel("Passenger Count")
ax.legend(["Passengers"])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot1_survival_dist.png"), dpi=300)
plt.close()

# -----------------------------
# Plot 2: Survival rate by Pclass and Sex
# -----------------------------
grouped = df.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=grouped, x="Pclass", y="Survived", hue="Sex")
ax.set_title("Survival Rate by Pclass and Sex")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate")
ax.legend(title="Sex")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot2_survival_by_pclass_sex.png"), dpi=300)
plt.close()

# -----------------------------
# Plot 3: Age distribution with survival overlay
# -----------------------------
plt.figure(figsize=(10, 6))
ax = sns.histplot(data=df, x="Age", hue="Survived", bins=30, multiple="layer", alpha=0.5)
ax.set_title("Age Distribution with Survival Overlay")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend(title="Survived")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot3_age_distribution.png"), dpi=300)
plt.close()

# -----------------------------
# Plot 4: Missing value heatmap
# -----------------------------
plt.figure(figsize=(12, 6))
ax = sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
ax.set_title("Missing Value Heatmap")
ax.set_xlabel("Columns")
ax.set_ylabel("Rows")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot4_missing_heatmap.png"), dpi=300)
plt.close()

# -----------------------------
# Summary
# -----------------------------
survival_rate = df["Survived"].mean()
class_counts = df["Survived"].value_counts()
missing = df.isnull().mean() * 100

summary = f"""
EDA Summary:

1. Class imbalance:
Survival rate = {survival_rate:.2%}. Counts: {class_counts.to_dict()}.

2. Key predictors:
Sex and Pclass show strong relationships with survival. Females and higher-class passengers have higher survival rates.
Age shows distribution differences between survivors and non-survivors.

3. Missing values and leakage:
Columns with missing values: {missing[missing > 0].to_dict()}.
Potential leakage risk exists in identifiers such as Name, Ticket, Cabin.

4. Outliers:
Fare and Age may contain outliers based on distribution spread.
"""

print(summary)

with open(os.path.join(output_dir, "eda_summary.txt"), "w") as f:
    f.write(summary)