# Causal-Learn API

This notebook demonstrates how to use the native Causal-Learn Python API to infer causal structure in multivariate time series. It covers:

- Loading and preprocessing financial time-series data  
- Computing Granger-causality tests  
- Running the PC algorithm from Causal-Learn  
- Visualizing the inferred causal graph  

Refer to the detailed API documentation in `causal_learn.API.md`. :contentReference[oaicite:0]{index=0}

Follow the Causify coding-style guide:  
https://github.com/causify-ai/helpers/blob/master/docs/coding/all.jupyter_notebook.how_to_guide.md

---

## Imports  
```python
# Load autoreload and plotting extensions
%load_ext autoreload
%autoreload 2
%matplotlib inline

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Causal-Learn imports
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

# Statsmodels for Granger tests
from statsmodels.tsa.stattools import grangercausalitytests

```
## Configuration
```python
import helpers.hdbg as hdbg
import helpers.hprint as hprint

# Initialize logger
hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)

# Notebook display settings
hprint.config_notebook()

```

## Load Data

```python
# 1. Read CSVs or fetch via yfinance
df_btc = pd.read_csv("data/BTC-USD.csv", parse_dates=["Date"])
df_sp = pd.read_csv("data/^GSPC.csv", parse_dates=["Date"])

# 2. Merge on Date
df = df_btc.merge(df_sp, on="Date", suffixes=("_btc", "_sp"))
```

## Compute Statistics Data

```python
# Compute daily log returns and volume changes
df["r_btc"] = np.log(df["Close_btc"]).diff()
df["r_sp"]  = np.log(df["Close_sp"] ).diff()
df["vol_pct"] = df["Volume_btc"].pct_change()

df = df.dropna().reset_index(drop=True)
```

## Clean Data

```python 
# Ensure no missing values remain
df = df[["Date", "r_btc", "r_sp", "vol_pct"]].dropna()

```

## Compute Statistics Again

```python
# Optional: summary statistics
df[["r_btc", "r_sp", "vol_pct"]].describe()
```
