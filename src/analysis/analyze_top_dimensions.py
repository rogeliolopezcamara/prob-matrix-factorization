import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Ensure src is in path if needed (though running as module is preferred)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def analyze_top_dimensions(model_name, n_dim, n_items):
    """
    Analyzes item embeddings to find dimensions with highest volatility (variance),
    and visualizes the top and bottom items for those dimensions.
    """
    
    # 1. Construct paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    embeddings_path = os.path.join(base_path, 'data', 'embeddings', model_name, 'item_embeddings.csv')
    raw_recipes_path = os.path.join(base_path, 'data', 'raw', 'RAW_recipes.csv')
    
    output_dir = os.path.join(base_path, 'reports', 'figures', 'Top_recepies_dim', model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading embeddings from: {embeddings_path}")
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file not found at {embeddings_path}")
        return

    # 2. Load Embeddings
    df_emb = pd.read_csv(embeddings_path)
    
    if 'recipe_id' not in df_emb.columns:
        print("Error: 'recipe_id' column missing in embeddings file.")
        return
        
    # Separate recipe_id and latent dimensions
    recipe_ids = df_emb['recipe_id']
    # Assuming all other columns are latent dimensions (except maybe unnamed indices if any)
    # Filter for numeric columns that look like dimensions (usually 0, 1, 2... or dim_0...)
    # We'll assume all non-id cols are dimensions.
    latent_cols = [c for c in df_emb.columns if c != 'recipe_id']
    X = df_emb[latent_cols]
    
    print(f"Loaded embeddings with shape: {df_emb.shape}. Found {len(latent_cols)} dimensions.")

    # 3. Calculate "Divergence" Score (Mean(Top N) - Mean(Bottom N))
    print(f"Calculating divergence scores for {len(latent_cols)} dimensions...")
    scores = {}
    for dim in latent_cols:
        # Get top n_items and bottom n_items values
        top_vals = df_emb[dim].nlargest(n_items)
        bot_vals = df_emb[dim].nsmallest(n_items)
        
        # Calculate score: "Relative Difference"
        score = top_vals.mean() - bot_vals.mean()
        scores[dim] = score

    # Convert to series for easy handling
    scores_series = pd.Series(scores)

    # 4. Select Top n_dim dimensions by Score
    top_dims = scores_series.nlargest(n_dim).index.tolist()
    print(f"Top {n_dim} dimensions by divergence: {top_dims}")
    
    # 5. Load Recipe Names
    print(f"Loading recipes from: {raw_recipes_path}")
    df_recipes = pd.read_csv(raw_recipes_path, usecols=['id', 'name'])
    
    # 6. Merge to get names
    # map recipe_id in embeddings to id in distinct recipes
    df_merged = df_emb.merge(df_recipes, left_on='recipe_id', right_on='id', how='left')
    
    # Fill NAs in name if any
    df_merged['name'] = df_merged['name'].fillna('Unknown Recipe')
    
    # 7. Generate Figures
    
    # Helper to clean text for plotting
    def clean_text(text_list):
        return "\n".join([f"- {t[:40]}..." if len(t) > 40 else f"- {t}" for t in text_list])

    # --- Plot Top Items ---
    fig_top, axes_top = plt.subplots(1, n_dim, figsize=(4 * n_dim, 6))
    if n_dim == 1: axes_top = [axes_top]
    
    for idx, dim in enumerate(top_dims):
        # Sort by dimension value descending
        top_items = df_merged.nlargest(n_items, dim)
        names = top_items['name'].tolist()
        vals = top_items[dim].tolist()
        
        ax = axes_top[idx]
        ax.set_title(f"Dim: {dim}\n(Div: {scores_series[dim]:.4f})", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        text_content = "TOP RECIPES:\n\n" + clean_text(names)
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, verticalalignment='top', fontsize=9)
    
    plt.suptitle(f"Top {n_items} Recipes for Top {n_dim} Divergent Dimensions ({model_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_top = os.path.join(output_dir, f'Top_{n_dim}_{n_items}.png')
    plt.savefig(save_path_top, dpi=300)
    print(f"Saved: {save_path_top}")
    plt.close()
    
    # --- Plot Bottom Items ---
    fig_bot, axes_bot = plt.subplots(1, n_dim, figsize=(4 * n_dim, 6))
    if n_dim == 1: axes_bot = [axes_bot]
    
    for idx, dim in enumerate(top_dims):
        # Sort by dimension value ascending
        bot_items = df_merged.nsmallest(n_items, dim)
        names = bot_items['name'].tolist()
        vals = bot_items[dim].tolist()
        
        ax = axes_bot[idx]
        ax.set_title(f"Dim: {dim}\n(Div: {scores_series[dim]:.4f})", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        text_content = "BOTTOM RECIPES:\n\n" + clean_text(names)
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, verticalalignment='top', fontsize=9)
        
    plt.suptitle(f"Bottom {n_items} Recipes for Top {n_dim} Divergent Dimensions ({model_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_bot = os.path.join(output_dir, f'Bottom_{n_dim}_{n_items}.png')
    plt.savefig(save_path_bot, dpi=300)
    print(f"Saved: {save_path_bot}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and visualize top dimensions of embeddings.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model (e.g., gaussian_mf)')
    parser.add_argument('--n_dim', type=int, required=True, help='Number of top volatile dimensions to analyze')
    parser.add_argument('--n_items', type=int, required=True, help='Number of items to show in top/bottom lists')
    
    args = parser.parse_args()
    
    analyze_top_dimensions(args.model, args.n_dim, args.n_items)
