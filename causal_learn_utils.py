"""

This module contains reusable utility functions and wrappers for:
- loading and preprocessing market data (Bitcoin & S&P 500)
- computing returns, volume changes, and rolling correlations
- running Granger causality tests
- invoking Causal-Learn for directed causal discovery
- plotting results (heatmaps, causal graphs)

Notebooks should import and call these functions instead of embedding
complex logic inline, keeping analysis clean and modular.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------
def load_data(df: pd.DataFrame,
              date_col: str = "Date",
              index_col: str = None) -> pd.DataFrame:
    """
    Ensure Date column is datetime, sort by date, and set index if specified.
    :param df: raw DataFrame with a date column
    :param date_col: name of the date column
    :param index_col: if provided, set this column as index after parsing dates
    """
    logger.info("Loading and preparing DataFrame")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    if index_col:
        df = df.set_index(index_col)
    return df

def compute_log_returns(series: pd.Series) -> pd.Series:
    """
    Compute daily log returns: log(s_t / s_{t-1})
    :param series: price series
    :return: log-return series
    """
    logger.info("Computing log returns")
    return np.log(series).diff().dropna()

def compute_volume_change(series: pd.Series) -> pd.Series:
    """
    Compute daily percentage change in volume.
    :param series: volume series
    :return: percent-change series
    """
    logger.info("Computing volume percent changes")
    return series.pct_change().dropna()

# -----------------------------------------------------------------------------
# Rolling Correlation & Heatmap
# -----------------------------------------------------------------------------
def rolling_correlation(df: pd.DataFrame,
                        cols: List[str],
                        window: int = 60) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations for specified columns.
    :param df: DataFrame with return/volume columns
    :param cols: list of column names to correlate
    :param window: rolling window size (days)
    :return: MultiIndex DataFrame of correlations
    """
    logger.info("Calculating rolling correlations with window=%d", window)
    roll = df[cols].rolling(window)
    corr_list = []
    dates = []
    for date, window_df in roll:
        if window_df.shape[0] == window:
            corr = window_df.corr().unstack()
            corr_list.append(corr)
            dates.append(date)
    corr_df = pd.DataFrame(corr_list, index=dates)
    corr_df.index.name = 'Date'
    return corr_df

def plot_heatmap(matrix: pd.DataFrame,
                 title: str,
                 cmap: str = "coolwarm",
                 vmin: float = -1.0,
                 vmax: float = 1.0) -> None:
    """
    Plot a heatmap of a square pandas DataFrame.
    :param matrix: square DataFrame (e.g., correlation or p-value matrix)
    :param title: plot title
    """
    logger.info("Plotting heatmap: %s", title)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Granger Causality
# -----------------------------------------------------------------------------
def granger_causality_matrix(df: pd.DataFrame,
                             maxlag: int = 4,
                             test: str = "ssr_chi2test") -> pd.DataFrame:
    """
    Compute Granger causality p-values for all pairs in df.
    :param df: DataFrame of time series columns
    :param maxlag: maximum lag to test
    :param test: test statistic ('ssr_chi2test', etc.)
    :return: DataFrame of p-values (rows = caused, cols = cause)
    """
    cols = df.columns
    pvals = pd.DataFrame(np.ones((len(cols), len(cols))),
                         index=cols, columns=cols)
    logger.info("Running Granger causality tests (maxlag=%d)", maxlag)
    for caused in cols:
        for cause in cols:
            if caused == cause:
                continue
            test_result = grangercausalitytests(
                df[[caused, cause]], maxlag=maxlag, verbose=False)
            pval = test_result[maxlag][0][test][1]
            pvals.loc[caused, cause] = pval
    return pvals

# -----------------------------------------------------------------------------
# Causal-Learn PC Algorithm
# -----------------------------------------------------------------------------
def infer_pc_graph(data: np.ndarray,
                   var_names: List[str],
                   alpha: float = 0.05,
                   max_cond_vars: int = 2) -> "Graph":
    """
    Run the PC algorithm on multivariate time-series data.
    :param data: 2D array shape (n_samples, n_vars)
    :param var_names: list of variable names
    :param alpha: significance level for conditional independence tests
    :param max_cond_vars: max conditioning set size
    :return: directed graph object
    """
    logger.info("Inferring causal graph using PC algorithm")
    data = pd.DataFrame(data, columns=var_names).values
    cg = pc(data, alpha, max_cond_vars)
    GraphUtils.to_nx_graph(cg.G, var_names)  # converts internals for plotting
    return cg.G

def plot_pc_graph(graph, title: str = "Causal Graph") -> None:
    """
    Plot a NetworkX-style causal graph inferred by Causal-Learn.
    :param graph: NetworkX DiGraph
    :param title: plot title
    """
    import networkx as nx
    logger.info("Plotting causal graph: %s", title)
    plt.figure(figsize=(8, 6))
    pos = nx.circular_layout(graph)
    nx.draw_networkx(graph, pos,
                     with_labels=True,
                     node_color="skyblue",
                     edge_color="navy",
                     arrowsize=12)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
