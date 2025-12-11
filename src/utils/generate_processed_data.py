import pandas as pd
import os

def generate_processed_data():
    raw_interactions_path = 'data/raw/RAW_interactions.csv'
    pp_recipes_path = 'data/raw/PP_recipes.csv'
    pp_users_path = 'data/raw/PP_users.csv' # Kept for reference, though not used for mapping
    
    # Paths for user mapping construction
    train_path = 'data/raw/interactions_train.csv'
    test_path = 'data/raw/interactions_test.csv'
    val_path = 'data/raw/interactions_validation.csv'
    
    output_dir = 'data/processed'
    output_file = os.path.join(output_dir, 'interactions_processed.csv')
    
    print("Loading datasets...")
    raw_interactions = pd.read_csv(raw_interactions_path)
    pp_recipes = pd.read_csv(pp_recipes_path)
    
    # Load separate interaction files to build user map
    print("Building user mapping from train/test/val sets...")
    interactions_train = pd.read_csv(train_path)
    interactions_test = pd.read_csv(test_path)
    interactions_val = pd.read_csv(val_path)
    
    all_interactions_mapped = pd.concat([interactions_train, interactions_test, interactions_val])
    
    # Create user_id -> u mapping
    # We drop duplicates to get unique user mappings
    user_map = all_interactions_mapped[['user_id', 'u']].drop_duplicates()
    
    # Check for duplicate mappings for the same user_id (shouldn't happen but good to verify)
    if user_map['user_id'].duplicated().any():
        print("Warning: Duplicate user_id mappings found. Using the first occurrence.")
        user_map = user_map.drop_duplicates(subset=['user_id'])
    
    # Create recipe_id -> i mapping from PP_recipes
    # PP_recipes has 'id' which corresponds to 'recipe_id' in interactions
    recipe_map = pp_recipes[['id', 'i']].rename(columns={'id': 'recipe_id'})
    
    print("Merging mappings...")
    # Merge u
    df = raw_interactions.merge(user_map, on='user_id', how='inner')
    # Merge i
    df = df.merge(recipe_map, on='recipe_id', how='inner')
    
    print(f"Interactions after mapping: {len(df)}")
    
    # Filter: Keep recipes with >= 10 reviews
    print("Filtering recipes with < 10 reviews...")
    recipe_counts = df.groupby('recipe_id').size()
    valid_recipes = recipe_counts[recipe_counts >= 10].index
    
    df_filtered = df[df['recipe_id'].isin(valid_recipes)].copy()
    
    print(f"Interactions after filtering: {len(df_filtered)}")
    print(f"Unique recipes: {df_filtered['recipe_id'].nunique()}")
    print(f"Unique users: {df_filtered['user_id'].nunique()}")
    
    # Select desired columns
    final_columns = ['user_id', 'recipe_id', 'date', 'rating', 'u', 'i']
    df_final = df_filtered[final_columns]
    
    os.makedirs(output_dir, exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")
    
    # Split into Train (80%), Validation (10%), Test (10%)
    print("Splitting data into Train (80%), Val (10%), Test (10%)...")
    # Shuffle the dataframe
    df_shuffled = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(df_shuffled)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train = df_shuffled.iloc[:train_end]
    val = df_shuffled.iloc[train_end:val_end]
    test = df_shuffled.iloc[val_end:]
    
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Saved splits: Train={len(train)}, Val={len(val)}, Test={len(test)}")

if __name__ == "__main__":
    generate_processed_data()
