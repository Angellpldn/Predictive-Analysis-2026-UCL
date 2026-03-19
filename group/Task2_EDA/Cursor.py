import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_titanic_csv():
    filename = "Titanic-Dataset.csv"
    candidates = []

    # Search from current working dir and script dir (limited to repo size).
    roots = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    for root in roots:
        pattern = os.path.join(root, "**", filename)
        matches = glob.glob(pattern, recursive=True)
        for m in matches:
            candidates.append(m)

    if not candidates:
        raise FileNotFoundError(
            "Could not find 'Titanic-Dataset.csv'. Searched recursively from "
            f"{os.getcwd()} and the script directory."
        )

    # Prefer shortest path (usually closer to the actual intended dataset location).
    candidates = sorted(set(candidates), key=lambda p: len(p))
    return candidates[0]

def get_col(df, wanted):
    for c in df.columns:
        if str(c).strip().lower() == wanted.strip().lower():
            return c
    return None

def save_close(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    dataset_path = find_titanic_csv()
    df = pd.read_csv(dataset_path)

    survived_col = get_col(df, "Survived")
    sex_col = get_col(df, "Sex")
    pclass_col = get_col(df, "Pclass")
    age_col = get_col(df, "Age")

    if survived_col is None:
        raise KeyError("Target column 'Survived' not found.")
    if sex_col is None:
        raise KeyError("Column 'Sex' not found.")
    if pclass_col is None:
        raise KeyError("Column 'Pclass' not found.")
    if age_col is None:
        raise KeyError("Column 'Age' not found.")

    # Coerce key columns to numeric where appropriate
    df[survived_col] = pd.to_numeric(df[survived_col], errors="coerce")
    df[pclass_col] = pd.to_numeric(df[pclass_col], errors="coerce")
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    # Drop rows with missing target for plotting/summary
    df_model = df.dropna(subset=[survived_col]).copy()
    df_model[survived_col] = df_model[survived_col].astype(int)

    out_dir = os.getcwd()

    # -------------------------
    # Plot 1: Survival rate distribution (bar chart)
    # -------------------------
    survived_counts = df_model[survived_col].value_counts().reindex([0, 1], fill_value=0)
    survived_rates = (survived_counts / survived_counts.sum()).reindex([0, 1], fill_value=0)

    fig1 = plt.figure(figsize=(7, 5))
    x_labels = ["Not Survived (0)", "Survived (1)"]
    bar_positions = np.arange(2)
    bar_heights = [survived_rates.loc[0], survived_rates.loc[1]]
    labels = ["Not Survived", "Survived"]

    for i in range(2):
        plt.bar(bar_positions[i], bar_heights[i], width=0.6, label=labels[i])

    plt.xticks(bar_positions, x_labels)
    plt.ylabel("Proportion")
    plt.title("Survival Rate Distribution")
    plt.ylim(0, max(0.05, float(max(bar_heights)) * 1.15))
    plt.legend(title="Survival", loc="best")

    save_close(fig1, os.path.join(out_dir, "plot1_survival_dist.png"))

    # -------------------------
    # Plot 2: Survival rate by Pclass and Sex (grouped bar chart)
    # -------------------------
    plot_df = df_model.dropna(subset=[pclass_col, sex_col]).copy()

    # Ensure clean categorical values
    plot_df[sex_col] = plot_df[sex_col].astype(str).str.strip()

    survival_by = (
        plot_df.groupby([pclass_col, sex_col])[survived_col]
        .mean()
        .reset_index(name="survival_rate")
    )

    # Build consistent ordering
    pclass_order = [1, 2, 3]
    sexes = sorted(plot_df[sex_col].dropna().unique().tolist())
    preferred = ["female", "male"]
    if all(s in sexes for s in preferred):
        sexes = preferred
    else:
        # Fallback: put the two most frequent labels first (if any)
        freq = plot_df[sex_col].value_counts()
        sexes = freq.index.tolist()

    # Pivot to rates
    pivot = survival_by.pivot(index=pclass_col, columns=sex_col, values="survival_rate")
    pivot = pivot.reindex(pclass_order)

    fig2 = plt.figure(figsize=(9, 5))
    group_positions = np.arange(len(pclass_order))
    total_groups = len(sexes)
    bar_width = 0.8 / max(1, total_groups)

    for j, s in enumerate(sexes):
        rates = pivot[s].reindex(pclass_order)
        if rates.isna().all():
            continue
        x = group_positions + j * bar_width - (0.8 / 2) + (bar_width / 2)
        plt.bar(x, rates.values, width=bar_width, label=str(s))

    plt.xticks(group_positions, [str(pc) for pc in pclass_order])
    plt.xlabel("Pclass")
    plt.ylabel("Survival Rate")
    plt.title("Survival Rate by Pclass and Sex")
    plt.ylim(0, 1)
    plt.legend(title="Sex", loc="best")

    save_close(fig2, os.path.join(out_dir, "plot2_survival_by_pclass_sex.png"))

    # -------------------------
    # Plot 3: Age distribution with survival overlay (histogram)
    # -------------------------
    age_plot_df = df_model.dropna(subset=[age_col, survived_col]).copy()
    age_min = float(age_plot_df[age_col].min()) if len(age_plot_df) else 0.0
    age_max = float(age_plot_df[age_col].max()) if len(age_plot_df) else 1.0
    if not np.isfinite(age_min) or not np.isfinite(age_max):
        age_min, age_max = 0.0, 1.0
    if age_min == age_max:
        age_min, age_max = age_min - 1.0, age_max + 1.0

    bins = 30
    bin_edges = np.linspace(age_min, age_max, bins + 1)

    fig3 = plt.figure(figsize=(9, 5))
    for s, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        subset = age_plot_df[age_plot_df[survived_col] == s][age_col].astype(float)
        plt.hist(
            subset,
            bins=bin_edges,
            density=True,
            alpha=0.55,
            label=f"Survived = {s}",
            color=color
        )

    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.title("Age Distribution with Survival Overlay")
    plt.legend(title="Survival", loc="best")

    save_close(fig3, os.path.join(out_dir, "plot3_age_dist_survival_overlay.png"))

    # -------------------------
    # Plot 4: Missing value heatmap
    # -------------------------
    missing = df.isna()
    cols_with_missing = missing.columns[missing.any()].tolist()

    # Heatmap of missingness for columns that contain missing values
    fig4 = plt.figure(figsize=(12, 3 + 0.35 * max(1, len(cols_with_missing))))
    if cols_with_missing:
        miss_mat = missing[cols_with_missing].astype(int).to_numpy()  # rows = rows, cols = features
        # We display transposed: features x row-index
        im = plt.imshow(miss_mat.T, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        cbar = plt.colorbar(im)
        cbar.set_label("Missing (1) / Not Missing (0)")

        plt.yticks(range(len(cols_with_missing)), cols_with_missing)
        plt.xticks([])
        plt.xlabel("Row index")
        plt.ylabel("Feature")
        plt.title("Missing Values Heatmap (Only Features With Missingness)")
    else:
        plt.text(0.5, 0.5, "No missing values found.", ha="center", va="center")
        plt.axis("off")
        plt.title("Missing Values Heatmap")

    save_close(fig4, os.path.join(out_dir, "plot4_missing_value_heatmap.png"))

    # -------------------------
    # Written summary (printed)
    # -------------------------
    total = len(df_model)
    c0 = int(survived_counts.loc[0])
    c1 = int(survived_counts.loc[1])
    p0 = (c0 / total * 100.0) if total else 0.0
    p1 = (c1 / total * 100.0) if total else 0.0

    # Key predictors based on computed survival rates (matches the plots)
    survival_by_sex = (
        plot_df.dropna(subset=[sex_col])
        .groupby(sex_col)[survived_col]
        .mean()
        .sort_values(ascending=False)
    )
    survival_by_pclass = (
        plot_df.dropna(subset=[pclass_col])
        .groupby(pclass_col)[survived_col]
        .mean()
        .sort_values(ascending=False)
    )

    interaction_rates = (
        plot_df.dropna(subset=[pclass_col, sex_col])
        .groupby([pclass_col, sex_col])[survived_col]
        .mean()
    )

    best_combo = interaction_rates.idxmax() if len(interaction_rates) else None
    best_rate = float(interaction_rates.max()) if len(interaction_rates) else float("nan")
    worst_combo = interaction_rates.idxmin() if len(interaction_rates) else None
    worst_rate = float(interaction_rates.min()) if len(interaction_rates) else float("nan")

    # Missing value concerns
    missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    missing_any = missing_pct[missing_pct > 0]
    top_missing = missing_any.head(5)

    # Outliers (Age) based on IQR rule
    age_series = df_model[age_col].dropna().astype(float)
    outlier_info = {"count": 0, "min_outlier": None, "max_outlier": None}
    if len(age_series) >= 4:
        q1 = age_series.quantile(0.25)
        q3 = age_series.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outliers = age_series[(age_series < lo) | (age_series > hi)]
        outlier_info["count"] = int(len(outliers))
        outlier_info["min_outlier"] = float(outliers.min()) if len(outliers) else None
        outlier_info["max_outlier"] = float(outliers.max()) if len(outliers) else None

    # Leakage risks
    # For typical Titanic dataset, there are no