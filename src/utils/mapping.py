import pandas as pd
import numpy as np
import os

def get_recipe_id_map(data_dir='data'):
    """
    Returns a numpy array where array[i_new] = recipe_id.
    
    Mapping chain:
    1. i_new (model index) -> i (raw interaction index) via data/processed/dict_i.csv
    2. i -> id (raw recipe id) via data/raw/PP_recipes.csv
    """
    dict_i_path = os.path.join(data_dir, 'processed', 'dict_i.csv')
    pp_recipes_path = os.path.join(data_dir, 'raw', 'PP_recipes.csv')
    
    if not os.path.exists(dict_i_path):
        print(f"Error: {dict_i_path} not found.")
        return None
    if not os.path.exists(pp_recipes_path):
        print(f"Error: {pp_recipes_path} not found.")
        return None
        
    print("Loading mapping files...")
    # 1. Load dict_i (i_new -> i)
    dict_df = pd.read_csv(dict_i_path)
    if 'i_new' not in dict_df.columns or 'i' not in dict_df.columns:
        print("Error: dict_i.csv must contain 'i_new' and 'i' columns")
        return None
        
    # 2. Load PP_recipes (i -> id)
    # PP_recipes might be large, usecols helps
    pp_df = pd.read_csv(pp_recipes_path, usecols=['id', 'i'])
    
    # 3. Merge
    # dict_df: [i, i_new]
    # pp_df: [id, i]
    merged = pd.merge(dict_df, pp_df, on='i', how='left')
    
    # Check for missing
    if merged['id'].isnull().any():
        print(f"Warning: {merged['id'].isnull().sum()} items have no matching recipe_id in PP_recipes")
        # fill with -1 or keep as NaN (which enforces float)
        merged['id'] = merged['id'].fillna(-1)
        
    # 4. Create Array
    # Ensure sorted by i_new to align with model embeddings (0, 1, 2... N-1)
    merged = merged.sort_values('i_new')
    
    # Max index
    max_idx = merged['i_new'].max()
    n_items = max_idx + 1
    
    # Initialize array
    # recipe_id is integer
    id_map = np.zeros(n_items, dtype=int)
    
    # Assign (use values to ensure alignment)
    # Filter to only valid i_new (should be all)
    # But safeguard against messy data
    valid = merged[merged['i_new'] >= 0]
    
    # If there are gaps in i_new, zeros will remain (id 0?)
    # Usually recipe_ids are large integers. 0 might be confusing if valid.
    # But usually 0 is not a recipe_id in these datasets (often start at 1 or hash).
    
    id_map[valid['i_new'].values] = valid['id'].astype(int).values
    
    print(f"Mapping loaded. {len(valid)} items mapped.")
    return id_map
