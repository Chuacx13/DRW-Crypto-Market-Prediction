# DRW Crypto Market Prediction

Predicting short-horizon price changes from **order-book and flow-based features**.  
This project implements training (feature engineering, feature selection, modeling) and a lightweight prediction class that reuses saved scaler/PCA objects and trained models.

---

## Overview

Kaggle Competition: [DRW Crypto Market Prediction](https://www.kaggle.com/competitions/drw-crypto-market-prediction/overview)

- **Challenge**: dataset includes masked and unmasked features and shuffled time ordering, making traditional time-based features and models impossible.
- **Goal**: build robust predictive models using microstructure features and ensemble methods.

---

## Techniques Used

- **Feature Engineering – Market Microstructure**

  - Liquidity imbalance, bid/ask depth & ratios, order flow participation, buy/sell pressure
  - Proxies: Kyle’s λ, execution quality, order toxicity, depth/volume ratios, momentum-style interactions

- **Feature Processing**

  - Standardization (`StandardScaler`)
  - Variance filtering and removal of highly correlated features
  - Predictive Power and Mutual Information
  - Stationarity checks (ADF / KPSS)
  - **PCA** for dimensionality reduction (auto-selected components)

- **Models**

  - Linear Regression, Ridge, LightGBM, XGBoost
  - **Stacking ensemble** for final prediction

- **Evaluation Metric**
  - Pearson correlation on train vs evaluation splits

---

## Results

| Model             | Train Corr | Eval Corr  |
| ----------------- | ---------- | ---------- |
| XGBoost           | 0.4722     | 0.0919     |
| Ridge (α=100)     | 0.2180     | 0.1250     |
| LightGBM          | 0.6424     | 0.0848     |
| Linear            | 0.1931     | 0.1299     |
| **Stacked Final** | **0.6630** | **0.0674** |

---
