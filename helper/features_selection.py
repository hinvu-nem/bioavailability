import numpy as np
import pandas as pd


def remove_noise_columns(df, threshold=0.95):
    """
    Drop near-constant columns: any column where a single value makes up
    more than `threshold` fraction of rows carries little information.
    """
    keep_cols = []
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True).iloc[0]
        if top_freq < threshold:
            keep_cols.append(col)
    return df[keep_cols]


def remove_correlation(df, threshold=0.70):
    """
    Drop one column from every pair of numeric columns whose absolute
    correlation exceeds `threshold`, keeping the first-seen column.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)
