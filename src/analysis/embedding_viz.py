
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import ast

def load_recipe_mappings():
    """
    Loads the mapping from internal index 'i' to recipe 'id' from PP_recipes.csv.
    """
    pp_file = "data/raw/PP_recipes.csv"
    if not os.path.exists(pp_file):
        raise FileNotFoundError(f"File not found: {pp_file}")
    
    # Read only necessary columns
    df = pd.read_csv(pp_file, usecols=['id', 'i'])
    # Create dictionary mapping i -> id
    return df.set_index('i')['id'].to_dict()

def load_recipe_tags():
    """
    Loads the mapping from recipe 'id' to 'tags' from RAW_recipes.csv.
    """
    raw_file = "data/raw/RAW_recipes.csv"
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"File not found: {raw_file}")
    
    # Read only necessary columns
    df = pd.read_csv(raw_file, usecols=['id', 'tags'])
    # Create dictionary id -> tags (as list)
    # Tags are stored as string representation of list, so we need to parse them later or now.
    # Parsing 200k+ rows might be slow, let's just return the df or dict and parse on demand/batch.
    return df.set_index('id')['tags'].to_dict()

def load_embeddings(model_dir):
    """
    Loads item embeddings from the specified model directory.
    Handles files with or without headers/indices robustly.
    """
    params_file = os.path.join(model_dir, "item_embeddings.csv")
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Embeddings file not found: {params_file}")
    
    # Check if first line looks like a header (0,1,2...) or data (float)
    with open(params_file, 'r') as f:
        first_line = f.readline().strip()
        
    parts = first_line.split(',')
    has_header = False
    try:
        # If first few tokens are integers 0, 1, it's likely a generated header
        if parts[0] == "0" and parts[1] == "1":
            has_header = True
    except:
        pass
        
    if has_header:
        # Header exists, so we let pandas infer it (usually row 0)
        # We do NOT use index_col=0 because the header implies column names, not index at col 0
        df = pd.read_csv(params_file)
    else:
        # No header, assume data starts at row 0
        df = pd.read_csv(params_file, header=None)
        
    return df

def reduce_dimensions(df, method, n_components):
    """
    Reduces dimensions using the specified method.
    """
    if method == 'random':
        if n_components > df.shape[1]:
             raise ValueError(f"Target dimension {n_components} larger than original {df.shape[1]}")
        # Random sample of columns
        sampled_cols = df.sample(n=n_components, axis=1, random_state=42)
        return sampled_cols
        
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(df)
        return pd.DataFrame(reduced, columns=[f'PC{i+1}' for i in range(n_components)], index=df.index)
        
    elif method == 'umap':
        # UMAP can be slow, subsample if too large
        if df.shape[0] > 10000:
            print(f"Subsampling to 10000 items for UMAP (original: {df.shape[0]})")
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
            
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)
        reduced = reducer.fit_transform(df_sample)
        return pd.DataFrame(reduced, columns=[f'UMAP{i+1}' for i in range(n_components)], index=df_sample.index)
        
    elif method == 'tsne':
        # t-SNE is very slow, especially with method='exact' (required for n>3)
        limit = 1000 if n_components > 3 else 10000
        
        if df.shape[0] > limit:
            print(f"Subsampling to {limit} items for t-SNE (original: {df.shape[0]})")
            df_sample = df.sample(n=limit, random_state=42)
        else:
            df_sample = df
            
        method_tsne = 'barnes_hut' if n_components < 4 else 'exact'
        print(f"Using t-SNE method: {method_tsne} for {n_components} components on {df_sample.shape[0]} items.")
        
        reducer = TSNE(n_components=n_components, random_state=42, method=method_tsne)
        reduced = reducer.fit_transform(df_sample)
        return pd.DataFrame(reduced, columns=[f'tSNE{i+1}' for i in range(n_components)], index=df_sample.index)
        
    else:
        raise ValueError(f"Unknown method: {method}")

def plot_grid(df_reduced, method, model_name, output_dir, hue_labels=None):
    """
    Creates a pairplot (square matrix grid) and saves it.
    """
    # Create the matrix plot
    # PairGrid or pairplot from seaborn is perfect for "matrix cuadrada con nxn plots"
    # "En cada cuadrante vas a hacer un plot de todas los itemes donde se plote una dimensión contra la otra."
    
    # Prepare dataframe for plotting
    plot_df = df_reduced.copy()
    
    # If we have labels, add them to the dataframe
    if hue_labels is not None:
        # Align using index (which corresponds to reset index in main)
        # This handles subsampling correctly because df_reduced has index subset
        plot_df['Category'] = hue_labels.loc[plot_df.index]
        
    plt.figure(figsize=(20, 20))
    
    # Use PairGrid
    g = sns.PairGrid(plot_df, hue='Category' if hue_labels is not None else None, corner=False)
    # The user preferred the original style which had scatterplots on diagonal (implicit or map)
    g.map(sns.scatterplot, s=10, alpha=0.5)
    
    if hue_labels is not None:
        g.add_legend()
    
    # Save
    save_dir = os.path.join(output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{method}.png")
    
    g.savefig(save_path)
    print(f"Saved {method} plot into {save_path}")
    plt.close()

def get_category(tags_str, target_tags):
    """
    Determines the category of a recipe based on its tags.
    """
    try:
        # data is stored as string representation of list
        tags = ast.literal_eval(tags_str)
    except:
        return "Other"

    matched_tags = [t for t in tags if t in target_tags]
    
    if len(matched_tags) == 0:
        return "Other" # Or "None" or whatever default
    elif len(matched_tags) == 1:
        return matched_tags[0]
    else:
        return "Multiple"

def main():
    parser = argparse.ArgumentParser(description="Visualize item embeddings with dimensionality reduction.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory containing embeddings.")
    parser.add_argument("--dim", type=int, default=7, help="Target dimension for reduction.")
    parser.add_argument("--tags", nargs='*', default=[], help="List of optional tags to color points by.")
    
    args = parser.parse_args()
    
    model_name = os.path.basename(os.path.normpath(args.model_dir))
    output_dir = "reports/figures/dimension_reduction"
    
    print(f"Loading embeddings from {args.model_dir}...")
    df = load_embeddings(args.model_dir)
    print(f"Loaded {df.shape[0]} items with {df.shape[1]} dimensions.")
    
    categories = None
    if args.tags:
        print(f"Tag filtering enabled: {args.tags}")
        print("Loading recipe mappings...")
        try:
            i_to_id = load_recipe_mappings()
            id_to_tags = load_recipe_tags()
            
            # Map index (which is likely 'i') to tags
            # We assume the index of df corresponds to 'i' in PP_recipes
            # If df.index are 0,1,2..., check if they match 'i'
            # The prompt says "utilizas la columna i que te dice el item correspondiente (empieza en 0)"
            # So df.index should be 'i'.
            
            cats = []
            for i in df.index:
                # df.index might be an integer or string depending on csv load. 
                # Let's ensure integer access if possible or safe get
                try:
                    i_val = int(i)
                except:
                    i_val = i
                
                recipe_id = i_to_id.get(i_val)
                if recipe_id is None:
                    cats.append("Unknown")
                    continue
                
                tags_str = id_to_tags.get(recipe_id)
                if tags_str is None:
                    cats.append("Unknown")
                    continue
                    
                cat = get_category(tags_str, args.tags)
                cats.append(cat)
            
            # Pass as Series with index matching df (which we will reset)
            categories = pd.Series(cats)
            
            # Show distribution
            print(f"Categorization complete. distribution:\n{categories.value_counts()}")
            
        except Exception as e:
            print(f"Error processing tags: {e}")
            print("Proceeding without coloring.")
            categories = None
    
    # Reset index to ensure unique integer identification for subsampling alignment
    df.reset_index(drop=True, inplace=True)
    if categories is not None:
        categories.index = df.index

    methods = ['random', 'pca', 'umap', 'tsne']
    
    for method in methods:
        print(f"Processing {method}...")
        try:
            reduced_df = reduce_dimensions(df, method, args.dim)
            plot_grid(reduced_df, method, model_name, output_dir, hue_labels=categories)
        except Exception as e:
            print(f"Error processing {method}: {e}")

if __name__ == "__main__":
    main()
