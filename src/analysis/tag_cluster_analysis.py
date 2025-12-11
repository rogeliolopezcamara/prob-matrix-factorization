
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.cluster import KMeans
import ast


def load_data(model_dir, data_dir, nrows=None):
    """
    Loads embeddings and recipe metadata.
    """
    # 1. Load Embeddings
    emb_path = os.path.join(model_dir, "item_embeddings.csv")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    print(f"Loading embeddings from {emb_path}...")
    df_emb = pd.read_csv(emb_path, nrows=nrows)
    
    # 2. Load PP_recipes for ID mapping (i -> id)
    # Always load full mapping to ensure we find the ids corresponding to the loaded embeddings
    pp_path = os.path.join(data_dir, "PP_recipes.csv")
    print(f"Loading PP_recipes from {pp_path}...")
    df_pp = pd.read_csv(pp_path, usecols=['id', 'i'])
    
    # 3. Load RAW_recipes for Tags (id -> tags)
    # Always load full tags to ensure coverage
    raw_path = os.path.join(data_dir, "RAW_recipes.csv")
    print(f"Loading RAW_recipes from {raw_path}...")
    df_raw = pd.read_csv(raw_path, usecols=['id', 'tags'])
    
    return df_emb, df_pp, df_raw

def process_tags(tag_str):
    try:
        return ast.literal_eval(tag_str)
    except:
        return []

def main():
    parser = argparse.ArgumentParser(description="Analyze item clusters by tags.")
    parser.add_argument("--model_dir", type=str, default="data/embeddings/poisson_mf_cavi", help="Path to model embeddings.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw data.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (limit rows).")
    args = parser.parse_args()

    nrows = 1000 if args.debug else None


    # Load Data
    df_emb, df_pp, df_raw = load_data(args.model_dir, args.data_dir, nrows)
    
    # Merge Data
    # df_emb index should correspond to 'i' in PP_recipes?
    # Usually item embeddings are ordered by internal index.
    # Let's assume df_emb index is 0..N-1 which matches 'i'.
    
    # Filter df_pp to only include items present in embeddings (if any mismatch)
    # But usually they match.
    
    # Create master DF
    # We join df_emb (index) with df_pp (on 'i')
    # Note: df_emb might have named index. Reset index to get 'i'.
    df_emb = df_emb.reset_index(drop=True)
    df_emb['i'] = df_emb.index
    
    print("Merging data...")
    merged = pd.merge(df_emb, df_pp, on='i', how='inner')
    merged = pd.merge(merged, df_raw, on='id', how='inner')
    
    print(f"Merged data shape: {merged.shape}")
    
    # Parse tags
    print("Parsing tags...")
    merged['tags_list'] = merged['tags'].apply(process_tags)
    
    # UMAP
    # From embedding_viz.py: n_components=?, random_state=42
    # User requested: "mismos parámetros que se utilizaron en src.analysis" (embedding_viz)
    # embedding_viz: n_neighbors default (15), min_dist default (0.1), metric='euclidean'
    # It also subsamples to 10000.
    
    if merged.shape[0] > 10000:
        print("Subsampling to 10000 items for UMAP (consistency with embedding_viz)...")
        # Sample but keep track of indices
        df_sample = merged.sample(n=10000, random_state=42).copy()
    else:
        df_sample = merged.copy()
        
    # Extract embedding columns (drop metadata)
    # Columns: i, id, tags, tags_list are metadata. Rest are embedding dims.
    # Assuming embedding cols are like '0', '1', etc from read_csv or whatever came from file.
    # safely drop non-numeric or known metadata cols
    meta_cols = ['i', 'id', 'tags', 'tags_list']
    X = df_sample.drop(columns=[c for c in meta_cols if c in df_sample.columns])
    
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(X)
    
    df_sample['x'] = embedding_2d[:, 0]
    df_sample['y'] = embedding_2d[:, 1]
    
    # KMeans Clustering (2 clusters)
    print("Clustering (K=2)...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    df_sample['cluster'] = kmeans.fit_predict(embedding_2d)
    
    # Analyze Tags
    print("Analyzing tags...")
    # Count tags in each cluster
    from collections import Counter
    
    c0 = df_sample[df_sample['cluster'] == 0]
    c1 = df_sample[df_sample['cluster'] == 1]
    
    cnt0 = Counter([tag for tags in c0['tags_list'] for tag in tags])
    cnt1 = Counter([tag for tags in c1['tags_list'] for tag in tags])
    
    # Normalize counts
    n0 = len(c0)
    n1 = len(c1)
    
    # Get all unique tags with sufficient frequency
    all_tags = set(list(cnt0.keys()) + list(cnt1.keys()))
    
    tag_scores = []
    min_count = 50 # Filter very rare tags
    
    for tag in all_tags:
        count0 = cnt0[tag]
        count1 = cnt1[tag]
        
        if count0 + count1 < min_count:
            continue
            
        p0 = count0 / n0
        p1 = count1 / n1
        
        # Score: absolute difference in proportion
        diff = p0 - p1
        tag_scores.append({
            'tag': tag,
            'p0': p0,
            'p1': p1,
            'diff': diff,
            'abs_diff': abs(diff)
        })
        
    df_tags = pd.DataFrame(tag_scores)
    df_tags = df_tags.sort_values('abs_diff', ascending=False)
    
    print("\nTop 20 Discriminative Tags:")
    print(df_tags[['tag', 'p0', 'p1', 'diff']].head(20).to_string(index=False))
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_sample, x='x', y='y', hue='cluster', palette='viridis', s=10, alpha=0.6)
    plt.title("UMAP Projection of Item Embeddings (K-Means Clusters)")
    
    output_dir = "reports/figures/tag_clusters"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "umap_clusters.png")
    plt.savefig(out_path)
    print(f"\nSaved plot to {out_path}")
    
    # Save Tag Analysis
    tag_out_path = os.path.join(output_dir, "discriminative_tags.csv")
    df_tags.to_csv(tag_out_path, index=False)
    print(f"Saved tag analysis to {tag_out_path}")

if __name__ == "__main__":
    main()
