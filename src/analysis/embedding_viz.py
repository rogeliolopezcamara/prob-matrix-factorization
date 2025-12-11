
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

def load_recipe_tags():
    """
    Loads the mapping from recipe 'id' to 'tags' from RAW_recipes.csv.
    """
    raw_file = "data/raw/RAW_recipes.csv"
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"File not found: {raw_file}")
    
    # Read only necessary columns
    df = pd.read_csv(raw_file, usecols=['id', 'tags'])
    return df.set_index('id')['tags'].to_dict()

def load_embeddings(model_dir):
    """
    Loads item embeddings from the specified model directory.
    Assumes item_embeddings.csv has 'recipe_id' as the first column/index.
    """
    params_file = os.path.join(model_dir, "item_embeddings.csv")
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Embeddings file not found: {params_file}")
    
    # Load with index_col=0 assuming recipe_id is the first column
    df = pd.read_csv(params_file, index_col=0)
        
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
    
    # Prepare dataframe for plotting
    plot_df = df_reduced.copy()
    
    # If we have labels, add them to the dataframe
    if hue_labels is not None:
        # Align using index 
        # hue_labels should be a Series indexed by recipe_id, same as plot_df
        plot_df['Category'] = hue_labels.loc[plot_df.index]
        
    plt.figure(figsize=(20, 20))
    
    # Use PairGrid
    g = sns.PairGrid(plot_df, hue='Category' if hue_labels is not None else None, corner=False)
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
        return "Other"
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
        print("Loading recipe tags...")
        try:
            # i_to_id no longer needed as df.index IS recipe_id
            id_to_tags = load_recipe_tags()
            
            cats = []
            for recipe_id in df.index:
                # df.index (recipe_id) should be integer based on previous checks
                # but might be read as float if there were NaNs (which shouldn't happen with our mapping)
                # Safeguard casting
                try:
                    rid = int(recipe_id)
                except:
                    rid = recipe_id
                
                tags_str = id_to_tags.get(rid)
                if tags_str is None:
                    cats.append("Unknown")
                    continue
                    
                cat = get_category(tags_str, args.tags)
                cats.append(cat)
            
            # Pass as Series with index matching df
            categories = pd.Series(cats, index=df.index)
            
            # Show distribution
            print(f"Categorization complete. distribution:\n{categories.value_counts()}")
            
        except Exception as e:
            print(f"Error processing tags: {e}")
            import traceback
            traceback.print_exc()
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
