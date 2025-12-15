import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = "viridis"

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_plot(filename):
    output_dir = "reports/figures/exploratory_analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved {filename}")

def load_data():
    print("Loading data...")
    # Raw Data
    raw_train = pd.read_csv("data/raw/interactions_train.csv")
    raw_valid = pd.read_csv("data/raw/interactions_validation.csv")
    raw_test = pd.read_csv("data/raw/interactions_test.csv")
    
    # Aggregating Raw Data
    df_raw = pd.concat([raw_train, raw_valid, raw_test], ignore_index=True)
    
    # Processed Data
    proc_train = pd.read_csv("data/processed/interactions_train.csv")
    proc_valid = pd.read_csv("data/processed/interactions_validation.csv")
    proc_test = pd.read_csv("data/processed/interactions_test.csv")
    
    # Standardizing split names for visualization
    proc_train['split_type'] = 'Train'
    proc_valid['split_type'] = 'Validation'
    proc_test['split_type'] = 'Test'
    
    df_proc = pd.concat([proc_train, proc_valid, proc_test], ignore_index=True)
    
    return df_raw, df_proc

def plot_ratings_distribution(df, rating_col, title, filename, hue=None):
    plt.figure(figsize=(10, 6))
    if hue:
        sns.countplot(data=df, x=rating_col, hue=hue, palette=PALETTE)
    else:
        sns.countplot(data=df, x=rating_col, palette=PALETTE)
    plt.title(title)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    save_plot(filename)

def plot_long_tail(data, xlabel, title, filename, color='blue'):
    # data is a Series of counts
    data_values = data.values
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(data_values)[::-1], color=color, linewidth=2)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title)
    plt.xlabel(f"{xlabel} (Rank)")
    plt.ylabel("Count (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    save_plot(filename)

def plot_activity_hist(data, xlabel, title, filename, color='purple', bins=50):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=bins, color=color, log_scale=(True, True))
    plt.title(title)
    plt.xlabel(f"{xlabel} (Log Scale)")
    plt.ylabel("Frequency (Log Scale)")
    save_plot(filename)

def analyze_raw(df_raw):
    print("\n--- Analyzing Raw Data ---")
    print(f"Total Raw Interactions: {len(df_raw)}")
    print(f"Unique Users: {df_raw['user_id'].nunique()}")
    print(f"Unique Recipes: {df_raw['recipe_id'].nunique()}")
    
    # 1. Ratings Distribution
    plot_ratings_distribution(df_raw, 'rating', 'Raw Data - Ratings Distribution', 'raw_ratings_dist.png')
    
    # 2. User Activity (Long Tail)
    user_counts = df_raw.groupby('user_id').size()
    plot_long_tail(user_counts, 'User', 'Raw Data - User Activity (Rank-Frequency)', 'raw_user_activity_long_tail.png')
    plot_activity_hist(user_counts, 'Number of Ratings per User', 'Raw Data - User Activity Distribution', 'raw_user_activity_hist.png', color='teal')

    # 3. Item Popularity (Long Tail)
    item_counts = df_raw.groupby('recipe_id').size()
    plot_long_tail(item_counts, 'Recipe', 'Raw Data - Recipe Popularity (Rank-Frequency)', 'raw_recipe_popularity_long_tail.png', color='orange')
    plot_activity_hist(item_counts, 'Number of Ratings per Recipe', 'Raw Data - Recipe Popularity Distribution', 'raw_recipe_popularity_hist.png', color='coral')

def analyze_processed(df_proc):
    print("\n--- Analyzing Processed Data ---")
    print(f"Total Processed Interactions: {len(df_proc)}")
    print(f"Unique Users (u): {df_proc['u'].nunique()}")
    print(f"Unique Items (i): {df_proc['i'].nunique()}")
    
    # 1. Ratings Distribution (Aggregated)
    plot_ratings_distribution(df_proc, 'rating', 'Processed Data - Ratings Distribution (All)', 'proc_ratings_dist_all.png')
    
    # 2. Ratings Distribution (By Split)
    plot_ratings_distribution(df_proc, 'rating', 'Processed Data - Ratings Distribution by Split', 'proc_ratings_dist_split.png', hue='split_type')
    
    # 3. User Activity (Aggregated)
    user_counts = df_proc.groupby('u').size()
    plot_long_tail(user_counts, 'User (u)', 'Processed Data - User Activity', 'proc_user_activity_long_tail.png')

    # 4. Item Popularity (Aggregated)
    item_counts = df_proc.groupby('i').size()
    plot_long_tail(item_counts, 'Item (i)', 'Processed Data - Item Popularity', 'proc_item_popularity_long_tail.png', color='orange')
    
    # 5. Split Counts
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_proc, x='split_type', palette=PALETTE)
    plt.title("Processed Data - Interactions per Split")
    plt.xlabel("Split")
    plt.ylabel("Count")
    save_plot('proc_split_counts.png')

def main():
    df_raw, df_proc = load_data()
    
    analyze_raw(df_raw)
    analyze_processed(df_proc)
    
    print("\nAnalysis Complete. Figures saved to reports/figures/exploratory_analysis/")

if __name__ == "__main__":
    main()
