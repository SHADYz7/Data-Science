# Bitcoin–Equity–Volume Causality Analysis Example

## Project description
This project constructs a modular pipeline to analyze causal links between Bitcoin returns, S&P 500 returns, and Bitcoin trading volume. We:
- **Ingest data** from Yahoo Finance (daily close prices and volumes).
- **Preprocess** by computing daily log-returns for BTC and S&P 500, and percent changes in BTC volume.
- **Test Granger causality** at a 4-day lag using statsmodels.
- **Discover directed causal structure** with the PC algorithm in Causal-Learn.
- **Visualize** results via rolling‐correlation heatmaps and inferred causal graphs.

## Table of Contents
- [Project description](#project-description)  
- [Data and Preprocessing](#data-and-preprocessing)  
- [Methodology](#methodology)  
- [Results and Visualization](#results-and-visualization)  
- [Conclusion](#conclusion)  
- [References](#references)  

## Hierarchy
The markdown hierarchy is enforced as follows:
1. **Level 1**: Project Title  
2. **Level 2**: Major sections (Project description, Data and Preprocessing, etc.)  
3. **Level 3**: Subsections within each major section  

## General Guidelines
- Describe in detail how each API was customized—e.g., which statsmodels functions powered the Granger tests and how Causal-Learn’s PC algorithm was invoked.  
- Cite all code and research sources in the **References** section below.  
- Use an automated linter or markdown‐TOC tool to keep the table of contents in sync.  
- Follow the coding-style guide:  
  https://github.com/causify-ai/helpers/blob/master/docs/coding/all.coding_style.how_to_guide.md  
- Name this file according to convention: `causal_learn.example.md`.  
