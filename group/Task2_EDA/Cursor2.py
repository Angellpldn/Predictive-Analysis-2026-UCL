import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load dataset (robust path find)
# -----------------------------
def locate_dataset(filename="Titanic-Dataset.csv"):
    # Try common relative paths first
    candidates = [
        "data/Titanic-Dataset.csv",
        "Titanic-Dataset.csv",
        "group/data/Titanic-Dataset.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Fallback: recursive search from current directory
    matches = glob.glob(f"**/{filename}", recursive=True)
    if matches:
        return sorted(matches, key=len)[0]

    raise FileNotFoundError(f"Could not find {filename} from current directory: {os.getcwd()}")

data_path = locate_dataset()
df = pd.read_csv(data_path)

# Standardize expected columns (case-insensitive support)
col_map = {c.lower().strip(): c for c in df.columns}
required = ["survived", "pclass", "sex", "age"]
for col in required:
    if col not in col_map:
        raise KeyError(f"Required column '{col}' not found in dataset columns: {list(df.columns)}")

survived_col = col_map["survived"]
pclass_col = col_map["pclass"]
sex_col = col_map["sex"]
age_col = col_map["age"]

# Ensure numeric types where needed
df[survived_col] = pd.to_numeric(df[survived_col], errors="coerce")
df[pclass_col] = pd.to_numeric(df[pclass_col], errors="coerce")
df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

# Drop rows without target for survival-related analysis
df_model = df.dropna(subset=[survived_col]).copy()
df_model[survived_col] = df_model[survived_col].astype(int)

# -----------------------------------------
# Plot 1: Survival rate distribution (bar)
# -----------------------------------------
survival_counts = df_model[survived_col].value_counts().reindex([0, 1], fill_value=0)
survival_rates = survival_counts / survival_counts.sum()

plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["Not Survived (0)", "Survived (1)"],
    [survival_rates.loc[0], survival_rates.loc[1]],
    label="Survival Rate"
)
plt.title("Survival Rate Distribution")
plt.xlabel("Survival Status")
plt.ylabel("Proportion")
plt.ylim(0, max(0.1, survival_rates.max() * 1.2))
plt.legend()
for b in bars:
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center")
plt.tight_layout()
plt.savefig("plot1_survival_dist.png", dpi=200)
plt.close()

# ----------------------------------------------------------
# Plot 2: Survival rate by Pclass and Sex (grouped bar chart)
# ----------------------------------------------------------
group_df = df_model.dropna(subset=[pclass_col, sex_col]).copy()
group_df[sex_col] = group_df[sex_col].astype(str).str.strip().str.lower()

survival_by_group = (
    group_df.groupby([pclass_col, sex_col])[survived_col]
    .mean()
    .reset_index(name="survival_rate")
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=survival_by_group,
    x=pclass_col,
    y="survival_rate",
    hue=sex_col
)
plt.title("Survival Rate by Pclass and Sex")
plt.xlabel("Passenger Class (Pclass)")
plt.ylabel("Survival Rate")
plt.ylim(0, 1.0)
plt.legend(title="Sex")
plt.tight_layout()
plt.savefig("plot2_survival_by_pclass_sex.png", dpi=200)
plt.close()

# ------------------------------------------------------
# Plot 3: Age distribution with survival overlay (hist)
# ------------------------------------------------------
age_df = df_model.dropna(subset=[age_col, survived_col]).copy()

plt.figure(figsize=(9, 5))
sns.histplot(
    data=age_df,
    x=age_col,
    hue=survived_col,
    bins=30,
    stat="density",
    common_norm=False,
    element="step",
    alpha=0.4
)
plt.title("Age Distribution with Survival Overlay")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend(title="Survived", labels=["0", "1"])
plt.tight_layout()
plt.savefig("plot3_age_dist_survival_overlay.png", dpi=200)
plt.close()

# ---------------------------------
# Plot 4: Missing value heatmap
# ---------------------------------
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis")
plt.title("Missing Value Heatmap")
plt.xlabel("Features")
plt.ylabel("Samples")
# Add legend-like colorbar labeling
cbar = plt.gca().collections[0].colorbar
cbar.set_label("Missingness (False=0, True=1)")
plt.tight_layout()
plt.savefig("plot4_missing_value_heatmap.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# Written summary: imbalance, predictors, missingness, risk
# ---------------------------------------------------------
total_n = len(df_model)
not_survived_n = int(survival_counts.loc[0])
survived_n = int(survival_counts.loc[1])
not_survived_pct = (not_survived_n / total_n * 100) if total_n else 0
survived_pct = (survived_n / total_n * 100) if total_n else 0

survival_by_sex = group_df.groupby(sex_col)[survived_col].mean().sort_values(ascending=False)
survival_by_class = group_df.groupby(pclass_col)[survived_col].mean().sort_index()

interaction = group_df.groupby([pclass_col, sex_col])[survived_col].mean()
best_group = interaction.idxmax() if len(interaction) else None
worst_group = interaction.idxmin() if len(interaction) else None

missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
missing_nonzero = missing_pct[missing_pct > 0]

# Outlier detection for Age using IQR
age_non_null = df_model[age_col].dropna()
if len(age_non_null) >= 4:
    q1 = age_non_null.quantile(0.25)
    q3 = age_non_null.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    age_outliers = age_non_null[(age_non_null < lower) | (age_non_null > upper)]
else:
    age_outliers = pd.Series(dtype=float)

print("EDA SUMMARY")
print("=" * 80)
print(f"1) Class imbalance (target = {survived_col}):")
print(f"   - Not survived (0): {not_survived_n}/{total_n} ({not_survived_pct:.1f}%)")
print(f"   - Survived (1):     {survived_n}/{total_n} ({survived_pct:.1f}%)")
if abs(not_survived_pct - survived_pct) > 10:
    print("   - The target is moderately imbalanced.")
else:
    print("   - The target is relatively balanced.")

print("\n2) Key predictors from plots:")
if len(survival_by_sex) >= 1:
    print("   - Survival rate by sex:")
    for k, v in survival_by_sex.items():
        print(f"     * {k}: {v:.3f}")
if len(survival_by_class) >= 1:
    print("   - Survival rate by class:")
    for k, v in survival_by_class.items():
        print(f"     * Pclass {int(k)}: {v:.3f}")
if best_group and worst_group:
    print(
        f"   - Highest survival subgroup: Pclass {int(best_group[0])}, Sex {best_group[1]} "
        f"({interaction.loc[best_group]:.3f})"
    )
    print(
        f"   - Lowest survival subgroup:  Pclass {int(worst_group[0])}, Sex {worst_group[1]} "
        f"({interaction.loc[worst_group]:.3f})"
    )

print("\n3) Missing values / leakage concerns:")
if len(missing_nonzero) > 0:
    print("   - Missingness detected in:")
    for col, pct in missing_nonzero.items():
        print(f"     * {col}: {pct:.1f}% missing")
else:
    print("   - No missing values detected.")
print(
    "   - Leakage risk: fit imputers/encoders on training data only, "
    "and avoid using identifiers/high-cardinality proxies (e.g., PassengerId, Ticket, Name, Cabin) blindly."
)

print("\n4) Outliers identified:")
if len(age_outliers) > 0:
    print(
        f"   - Age outliers (IQR rule): {len(age_outliers)} rows "
        f"(min={age_outliers.min():.2f}, max={age_outliers.max():.2f})"
    )
else:
    print("   - No age outliers found by IQR rule (or insufficient non-missing age values).")

print("\nSaved plot files:")
print(" - plot1_survival_dist.png")
print(" - plot2_survival_by_pclass_sex.png")
print(" - plot3_age_dist_survival_overlay.png")
print(" - plot4_missing_value_heatmap.png")