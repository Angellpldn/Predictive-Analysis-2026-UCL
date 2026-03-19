import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path="titanic.csv"):
    try:
        # Assuming the dataset is available locally
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Local dataset not found. Loading seaborn's Titanic dataset as fallback...")
        df = sns.load_dataset("titanic")
        # Standardize column names if seaborn dataset is used
        df.rename(columns={
            "pclass": "Pclass", 
            "sex": "Sex", 
            "age": "Age", 
            "survived": "Survived"
        }, inplace=True)

    sns.set_theme(style="whitegrid")

    # 1. Survival rate distribution (bar chart)
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Survived', palette='pastel')
    plt.title('Survival Rate Distribution')
    plt.xlabel('Survived (0 = No, 1 = Yes)')
    plt.ylabel('Passenger Count')
    plt.savefig('plot1_survival_dist.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Survival rate by Pclass and Sex (grouped bar chart)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', palette='deep', errorbar=None)
    plt.title('Survival Rate by Passenger Class and Sex')
    plt.xlabel('Passenger Class (Pclass)')
    plt.ylabel('Average Survival Rate')
    plt.legend(title='Sex')
    plt.savefig('plot2_survival_by_pclass_sex.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Age distribution with survival overlay (histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30, palette='Set2')
    plt.title('Age Distribution with Survival Overlay')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('plot3_age_survival_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Missing value heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Value Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Rows (Passengers)')
    plt.savefig('plot4_missing_value_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    perform_eda()
