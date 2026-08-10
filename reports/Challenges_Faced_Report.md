## Report on Challenges Faced

**1. High proportion and structure of missing data**
Several columns (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, and the garage/basement quality fields) had large fractions of missing values. Naively dropping these columns or rows would have discarded most of the dataset and thrown away real signal, since — per the data dictionary — `NaN` in these columns encodes "feature does not exist" rather than "value unknown." *Technique used:* domain-informed imputation — `'None'` for absent categorical features, `0` for the corresponding numeric features (e.g. `GarageArea` = 0 when there is no garage), rather than generic mean/mode imputation which would have implied the feature existed at an average level.

**2. `LotFrontage` missingness not at random**
`LotFrontage` was missing for ~17% of rows without an obvious "doesn't exist" interpretation (every house fronts *some* street). *Technique used:* imputed with the median `LotFrontage` of the house's own `Neighborhood`, since frontage is strongly tied to local lot layout conventions — a plain global median would have ignored this structure.

**3. Mixed-type / mis-typed features**
`MSSubClass` is stored as an integer but is actually a categorical dwelling-type code (the numeric ordering is meaningless). *Technique used:* explicitly cast to string/categorical before modeling, so it is one-hot encoded rather than treated as a continuous magnitude.

**4. Skewed target variable**
`SalePrice` was right-skewed (skew ≈ 1.9), violating the normally-distributed-residual assumption that linear models rely on and letting a few very expensive homes dominate squared-error loss. *Technique used:* modeled `log1p(SalePrice)` throughout, and back-transformed predictions with `expm1` for dollar-scale reporting/evaluation — this both normalized the distribution and made percentage errors (rather than absolute dollar errors) the effective optimization target.

**5. Outliers**
A small number of homes had very large `GrLivArea` but unexpectedly low `SalePrice` (documented as data anomalies for this dataset, e.g. sales to family members or distressed sales). Left in, they disproportionately pulled linear model fits. *Technique used:* identified via scatterplot against the target and removed the ~2 clearest cases rather than aggressively filtering on any single feature, to avoid discarding legitimate high-end sales.

**6. High cardinality and imbalanced categoricals**
`Neighborhood` has 25 categories; features like `Utilities` and `Street` are almost constant (near-zero variance) and contribute little. *Technique used:* one-hot encoding via `ColumnTransformer` (with `handle_unknown='ignore'` for robustness to unseen categories at inference time) for genuinely informative categoricals, and manual removal of near-constant columns identified during EDA to reduce noise/dimensionality.

**7. Multicollinearity among size-related features**
`GarageCars` and `GarageArea`, and `TotalBsmtSF`/`1stFlrSF`/`GrLivArea`, are highly correlated with each other. This inflates coefficient variance in linear models even though it barely affects tree-based models. *Technique used:* retained the features (since tree-based models handle collinearity natively) but flagged this explicitly for anyone consuming the linear model's coefficients, and relied on regularization (Ridge/Lasso) to stabilize the linear fits.

**8. Risk of data leakage in preprocessing**
Median/mode imputation and scaling must be fit only on training data to give an honest estimate of test performance. *Technique used:* all preprocessing was wrapped in a `scikit-learn` `Pipeline`/`ColumnTransformer` so that `fit` happens only inside cross-validation folds and on the training split, never on the full dataset before splitting.

**9. Choosing an evaluation metric that matches the business goal**
Raw RMSE in dollars over-weights errors on expensive homes. *Technique used:* trained and primarily evaluated on log-scale RMSE (equivalent to the competition's own RMSLE-style metric), while also reporting back-transformed dollar RMSE/MAE for stakeholder-facing interpretability.

**10. Balancing model complexity against interpretability**
Boosting ensembles gave the best raw accuracy but are harder to explain to a non-technical buyer or stakeholder than a linear model's coefficients. *Technique used:* kept both an interpretable (Ridge/Lasso) and a high-accuracy (tuned boosting) model in the comparison, and used feature importance/coefficient plots to make the "black box" model's key drivers explainable even when it isn't fully transparent.