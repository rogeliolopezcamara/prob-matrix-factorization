# Probabilistic Matrix Factorization for Recipe Recommendation

This repository contains various probabilistic matrix factorization models for recipe recommendation.

## 1. Downloading and Extracting Data

To get started, you need to download and unzip the dataset. Run the following scripts:

```bash
python src/download_data.py
python src/unzip_data.py
```

## 2. Data Preprocessing

The original dataset split defined in the Kaggle source was not ideal for our purposes. It contained many recipes in the validation and test sets that were not present in the training set (cold-start problem). Additionally, there were many recipes with very few reviews, creating significant sparsity.

To address this, we implemented a filtering process to remove these sparse recipes and then generated a new train/validation/test split.

To generate the processed CSV files used for modeling, run:

```bash
python src/data/load_data.py
```

## 3. Hyperparameter Tuning

We conducted several experiments to select the optimal hyperparameters for each model.

*   **K-Search**: Scripts like `src/experiments/run_[model]_best_k.py` were used to find the best latent dimension size ($K$).
*   **Comprehensive Tuning**: The script `src/experiments/tune_all_models.py` runs a comprehensive hyperparameter search.

When you run `tune_all_models.py`, it generates a file named `best_hyperparams.txt`.

**Important:**
*   This `best_hyperparams.txt` file is used by all subsequent scripts (comparison and training).
*   You can manually modify `best_hyperparams.txt` to adjust configurations.
*   **However**, be aware that re-running `src/experiments/tune_all_models.py` will overwrite this file.

## 4. Comparing Models

To compare the performance of different models using the selected hyperparameters, run:

```bash
python src/experiments/compare_models.py
```

This script reads the configurations from `best_hyperparams.txt` and evaluates the models.

## 5. Training Final Models

Once you have compared the models and are satisfied with the hyperparameters, you can train the models on the dataset.

**Usage:**

```bash
python src/experiments/train_all_models.py --dataset_mode [MODE]
```

**Arguments:**
*   `--dataset_mode`: Specifies which dataset splits to use for training.
    *   `train`: Train primarily on the training set (default).
    *   `train+val`: Train on the combined training and validation sets.
    *   `full`: Train on the entire dataset (train + validation + test).

**Example:**
```bash
python src/experiments/train_all_models.py --dataset_mode full
```

This script will take the hyperparameters from `best_hyperparams.txt` and train the models.

## 6. Analysis and Visualization

We provide tools to analyze the learned embeddings and dimensions. These scripts require specific arguments to run.

### Analyze Top Dimensions
This script analyzes the top dimensions of the latent space to understand what features the model is capturing (volatility analysis).

**Usage:**
```bash
python src/analysis/analyze_top_dimensions.py --model [MODEL_NAME] --n_dim [N] --n_items [M]
```

**Arguments:**
*   `--model`: The name of the model folder in `data/embeddings/` (e.g., `gaussian_mf`, `poisson_mf`, `hpf_cavi`).
*   `--n_dim`: The number of top volatile dimensions to analyze (e.g., `5`).
*   `--n_items`: The number of items (recipes) to list as "top" and "bottom" for each dimension (e.g., `10`).

**Example:**
```bash
python src/analysis/analyze_top_dimensions.py --model gaussian_mf --n_dim 5 --n_items 10
```

### Embedding Visualization
This script generates visualizations of the embeddings to explore the structure of the learned latent space (e.g., clustering of similar recipes) using dimensionality reduction techniques (PCA, t-SNE, UMAP).

**Usage:**
```bash
python src/analysis/embedding_viz.py --model_dir [PATH_TO_EMBEDDINGS] --dim [D] --tags [TAG1] [TAG2] ...
```

**Arguments:**
*   `--model_dir`: Path to the model directory containing `item_embeddings.csv` (e.g., `data/embeddings/gaussian_mf`).
*   `--dim`: The target dimension for the reduction (default: `7`). The script produces an NxN grid of plots.
*   `--tags`: (Optional) A list of tags to color the points by. Use this to see if the embeddings separate recipes with specific tags (e.g., `vegan`, `dessert`).

**Example:**
```bash
python src/analysis/embedding_viz.py --model_dir data/embeddings/gaussian_mf --dim 5 --tags vegan dessert
```
