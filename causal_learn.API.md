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

# install causal-learn if needed
!pip install causallearn statsmodels seaborn --quiet
!pip install yfinance causal-learn statsmodels

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# causal-learn imports
from causallearn.search.ConstraintBased.PC import pc   # time-series PC algorithm
from causallearn.utils.GraphUtils import GraphUtils

# statsmodels for VAR & Granger
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

# aesthetic defaults
sns.set_context('talk')
plt.rcParams['figure.figsize'] = (12, 6)

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
# define tickers and period
tickers = ['BTC-USD', '^GSPC']
data = yf.download(tickers, period='2y', interval='1d', group_by='ticker', progress=False)

# assemble into single DataFrame
btc = data['BTC-USD'][['Close','Volume']].rename(columns={'Close':'BTC_Close','Volume':'BTC_Vol'})
sp500 = data['^GSPC']['Close'].rename('SP500_Close')
df = pd.concat([btc, sp500], axis=1).dropna()
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
