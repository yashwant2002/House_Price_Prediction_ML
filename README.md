# House Price Prediction — Ames, Iowa (PRCP-1020)

Predicting residential home sale prices in Ames, Iowa using the Kaggle
["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
dataset — 1,460 home sales described by 79 explanatory variables.

## Project overview

This project delivers a complete, end-to-end regression pipeline:

1. **Data analysis report** — full EDA on the raw dataset.
2. **Machine learning model** — a tuned model that predicts `SalePrice` and explains which
   features drive it.
3. **Buyer recommendations** — practical guidance for choosing a home by area, price, and
   requirements.
4. **Model comparison report** — multiple regression algorithms benchmarked, with a
   production recommendation.
5. **Challenges report** — data issues encountered and the techniques used to resolve them.

All analysis lives in a single Jupyter notebook, `House_Price_Prediction.ipynb`, with an
accompanying `report.pdf` write-up.

## Repository structure

```
.
├── data/
│   └── raw/
│       └── data.csv                 # raw training data (1460 rows × 81 columns)
├── images/
│   ├── eda_plots/                   # saved EDA figures (distributions, correlations, etc.)
│   └── model_performance_plots/     # CV/test performance charts
├── models/
│   └── best_model.pkl               # pickled best-performing pipeline (preprocessing + model)
├── notebooks/
│   └── House_Price_Prediction.ipynb # full analysis, modeling, and reporting notebook
├── report.pdf                       # rendered data analysis / modeling report
└── README.md
```

> Adjust the paths above to match your actual layout — the notebook currently reads data via
> `../data/raw/data.csv` and saves figures/models via `../images/...` and `../models/...`,
> i.e. it expects to be run from a `notebooks/` subfolder.

## Dataset

- **Rows:** 1,460 home sales
- **Columns:** 79 features + `Id` + target `SalePrice`
- **Feature types:** 36 numerical, 43 categorical
- **Target:** `SalePrice` — right-skewed (skew ≈ 1.88), modeled as `log1p(SalePrice)`

Full column definitions are in the original dataset documentation
(`data_description.txt` from the Kaggle competition).

## Setup

```bash
# Clone / download the project, then install dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost jupyter
```

Place the raw CSV at `data/raw/data.csv` (or update the path in the notebook's load cell),
then open and run the notebook:

```bash
jupyter notebook notebooks/House_Price_Prediction.ipynb
```

## Pipeline summary

| Stage | What happens |
|---|---|
| **Data cleaning** | Structural missing values (no pool/alley/fireplace/garage/basement) filled with `'None'`/`0`; `LotFrontage` imputed by neighborhood median; single `Electrical` value imputed with the mode. Zero missing values remain afterward. |
| **EDA** | Target distribution & log-transform, univariate (numerical + categorical), correlation analysis, bivariate (numerical vs. numerical, categorical vs. numerical), multivariate pairplots. Two extreme outliers (large `GrLivArea`, low `SalePrice`) removed. |
| **Feature engineering** | `HouseAge`, `RemodAge`, `TotalSF`, `TotalBath`, and other engineered features; ordinal quality/condition scales mapped to numeric 0–5. |
| **Preprocessing** | `ColumnTransformer` + `Pipeline` — median imputation + scaling for numeric features, most-frequent imputation + one-hot encoding for categorical features. Fit only on the training split to avoid leakage. |
| **Modeling** | 11 regression algorithms compared via 5-fold cross-validation (Linear/Ridge/Lasso/ElasticNet, Decision Tree, KNN, Random Forest, Gradient Boosting, AdaBoost, SVR, XGBoost). |
| **Tuning** | `RandomizedSearchCV` on the top boosting model (XGBoost). |
| **Evaluation** | RMSE and R² reported on both the log scale (optimization target) and back-transformed dollar scale (for interpretability). |

## Results

Best model on the held-out test set (20% split, 292 rows): **tuned XGBoost**

| Metric | Value |
|---|---|
| Test RMSE (log) | 0.1148 |
| Test R² | 0.922 |
| Test RMSE ($) | ~$18,334 |
| Test MAE ($) | ~$13,508 |

Regularized linear models (ElasticNet, Lasso) came a close second and remain a strong,
more interpretable alternative when explainability matters more than the last few percent
of accuracy.

See `report.pdf` for the full write-up: EDA figures, the complete model comparison table,
feature-importance analysis, buyer recommendations by neighborhood/quality/budget, and a
detailed account of the data-quality challenges encountered.

## Key findings

- **`OverallQual`, `GrLivArea`, `GarageCars`/`GarageArea`, `TotalBsmtSF`, and `YearBuilt`**
  are the strongest single predictors of `SalePrice`.
- **`Neighborhood`** carries strong location-based price signal independent of the house's
  physical attributes.
- Price is driven by interactions between features (size × quality × location), not any
  single variable in isolation.

## Model artifact

The best-performing pipeline (preprocessing + model) is saved to `models/best_model.pkl`
via `pickle`, ready to load for inference:

```python
import pickle

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# model expects a DataFrame with the same raw feature columns used in training
predictions_log = model.predict(new_data)
predictions_dollars = np.expm1(predictions_log)
```

## License / data source

Dataset courtesy of the Kaggle "House Prices: Advanced Regression Techniques" playground
competition, based on the Ames Housing dataset compiled by Dean De Cock.