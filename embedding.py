
import pandas as pd
import sys
import re
import numpy as np
from pathlib import Path
from typing import Optional
local_python_path = str(Path(__file__).parents[0])
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
from utils.bedrock_utils import embed_texts_from_series
from utils.plotly_utils import fix_and_write, HTML, write_excel
from datetime import datetime
from sklearn.decomposition import PCA
import plotly.express as px
import json
logger = get_logger(__name__)
config = load_config(config_path=Path(local_python_path)/ 'config.json', add_date=False)


def clean_ocr_for_embedding(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Text must be a string")

    # 1. Remove clearly garbage lines first
    lines = text.split('\n')
    filtered_lines = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        letters = sum(ch.isalpha() for ch in stripped)
        if letters / max(len(stripped), 1) < 0.4:
            # looks like mostly symbols/numbers → drop
            continue
        filtered_lines.append(stripped)

    text = "\n".join(filtered_lines)

    # 3. Fix hyphenation
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # 4. Normalize whitespace
    text = text.replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def embed_text_series(text_series: pd.Series, 
                       embedding_model_id: str = "text-embedding-3-large",
                       batch_size: int = 100) -> np.ndarray:
    """
    Embed texts from a pandas Series using the bedrock_utils embedding function.
    
    Args:
        text_series: pandas Series containing text strings to embed
        embedding_model_id: Embedding model ID to use
        batch_size: Number of texts to embed in each batch
        
    Returns:
        numpy array of shape (n_texts, embedding_dim) containing embeddings
    """
    return embed_texts_from_series(
        text_series=text_series,
        embedding_model_id=embedding_model_id,
        batch_size=batch_size
    )


def reduce_embeddings_to_2d(embeddings: np.ndarray, 
                            n_components: int = 2,
                            random_state: int = 42) -> np.ndarray:
    """
    Reduce embeddings to 2D using PCA.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features) containing embeddings
        n_components: Number of components for PCA (default: 2)
        random_state: Random state for reproducibility
        
    Returns:
        numpy array of shape (n_samples, n_components) containing 2D embeddings
        
    Example:
        >>> embeddings = np.random.rand(100, 1536)  # Example embeddings
        >>> embeddings_2d = reduce_embeddings_to_2d(embeddings)
        >>> print(embeddings_2d.shape)  # (100, 2)
    """
    if embeddings.shape[0] < n_components:
        raise ValueError(f"Number of samples ({embeddings.shape[0]}) must be >= n_components ({n_components})")
    
    pca = PCA(n_components=n_components, random_state=random_state)
    embeddings_2d = pca.fit_transform(embeddings)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA reduction: {embeddings.shape[1]}D -> {n_components}D")
    logger.info(f"Explained variance: {explained_variance:.2%}")
    
    return embeddings_2d


def plot_embeddings_2d(embeddings_2d: np.ndarray,
                       labels: Optional[pd.Series] = None,
                       title: str = "2D Embedding Visualization"):
    """
    Plot 2D embeddings using plotly.
    
    Args:
        embeddings_2d: numpy array of shape (n_samples, 2) containing 2D embeddings
        labels: Optional pandas Series with labels for each point (for coloring)
        title: Title for the plot
        output_path: Optional path to save the plot as HTML file
        
    Example:
        >>> embeddings_2d = np.random.rand(100, 2)
        >>> labels = pd.Series(['A'] * 50 + ['B'] * 50)
        >>> plot_embeddings_2d(embeddings_2d, labels=labels)
    """
    if embeddings_2d.shape[1] != 2:
        raise ValueError(f"embeddings_2d must have 2 columns, got {embeddings_2d.shape[1]}")
    
    df_plot = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    })
    
    if labels is not None:
        if len(labels) != len(embeddings_2d):
            raise ValueError(f"labels length ({len(labels)}) must match embeddings length ({len(embeddings_2d)})")
        df_plot['label'] = labels.values
    
    # Create plotly scatter plot
    if labels is not None:
        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color='label',
            title=title,
            labels={'x': 'PC1', 'y': 'PC2'}
        )
    else:
        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            title=title,
            labels={'x': 'PC1', 'y': 'PC2'}
        )
    
    return fig

def read_data():
    input_dir = Path(config["input_dir"]) / "American Stories" / 'clean'
    csv_files = list(input_dir.glob("*.csv"))
    
    
    # Read and concatenate all dataframes
    logger.info(f"Reading {len(csv_files)} CSV files...")
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]    
    logger.info(f"Read {len(dfs)} dataframes, concatenating...")
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Concatenated {len(df_all)} total rows from {len(csv_files)} files")
    return df_all
    

def main(sample_size: int = 400000, 
         embedding_model_id: str = "text-embedding-3-small"):
    """
    Main function to read CSVs, sample, embed, save embeddings, reduce dimensions, and plot.
    
    Args:
        sample_size: Number of rows to sample from the concatenated dataframes
        text_column: Name of the column containing text to embed
        embeddings_output_dir: Directory to save individual embedding files (defaults to config output_dir / 'embeddings')
    """
    # Read all CSV files from the clean directory
    df_all = read_data()
    
    logger.info(f"Total rows: {len(df_all)}")
    df_sample = df_all.sample(n=sample_size, random_state=42).reset_index(drop=True)
    logger.info(f"Sampled {len(df_sample)} rows")
    df_sample = df_sample[(df_sample['article_id'].notna()) & (df_sample['clean_article'] != '')].reset_index(drop=True)   
    logger.info(f"After cleaning, {len(df_sample)} rows remain")
    
        
    # Embed texts
    logger.info("Embedding texts...")
    df_sample['embeddings'] = embed_text_series(
        df_sample['clean_article'],
        embedding_model_id=embedding_model_id
    )
    
    logger.info(f"Saving embeddings to {Path(config["input_dir"]) / "American Stories" / "embeddings"}...")
    write_excel(df_sample, 'embeddings', output_dir=Path(config["input_dir"]) / "American Stories" / "embeddings")
    
    
    # Reduce dimensions to 2D
    logger.info("Reducing embeddings to 2D using PCA...")
    df_sample['embeddings_2d'] = reduce_embeddings_to_2d(df_sample['embeddings'])
    
    logger.info(f"Saving embeddings to {Path(config["input_dir"]) / "American Stories" / "embeddings_2d"}...")
    write_excel(df_sample, 'embeddings_2d', output_dir=Path(config["input_dir"]) / "American Stories" / "embeddings_2d")
    
    # Create plot
    logger.info("Creating plot...")
    fig = plot_embeddings_2d(
        df_sample['embeddings_2d'],
        labels=df_sample['article_id'],
        title=f"2D Embedding Visualization (n={len(df_sample)})"
    )
    
    # Save plot using fix_and_write
    fix_and_write(
        fig,
        filename="embeddings_2d_visualization",
        output_type=HTML
    )
    
    logger.info("Main function completed successfully")


if __name__ == "__main__":
    
    main( )