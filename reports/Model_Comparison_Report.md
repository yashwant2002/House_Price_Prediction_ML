## Model Comparison Report

**Summary of model performance (5-fold CV on training data, then confirmed on held-out 20% test set):**

| Aspect | Finding |
|---|---|
| Linear models (Linear / Ridge / Lasso / ElasticNet) | Fast, interpretable, competitive after regularization; Lasso/ElasticNet also perform implicit feature selection. Sensitive to multicollinearity and outliers even after log-transform. |
| Tree-based single models (Decision Tree, KNN) | Underperform — a single tree overfits or underfits depending on depth; KNN is hurt by the high-dimensional, mixed-type feature space. |
| Bagging ensembles (Random Forest) | Solid, robust performance with little tuning; less prone to overfitting than a single tree; not the best model class in this instance but a strong baseline. |
| Boosting ensembles (Gradient Boosting, AdaBoost, XGBoost, LightGBM) | Consistently the strongest performers on this tabular dataset. Handle non-linearities and interactions between features (e.g. quality × size) natively. |
| Support Vector Regression | Competitive but requires careful scaling and kernel/parameter choice; slower to train on this dataset than boosting methods. |

**Recommendation for production:** the results above (see `final_summary`) identify the best model empirically for this run. In practice for this dataset, gradient-boosted tree models (XGBoost / LightGBM / GradientBoosting) typically edge out linear models and Random Forest on both CV and held-out RMSE, while Ridge/Lasso remain attractive when interpretability or fast retraining matter more than the last few percent of accuracy.

**Practical guidance:**
- If **interpretability** is the priority (e.g. explaining valuations to stakeholders or regulators) → **Ridge/Lasso** regression on the log target, since coefficients are directly interpretable as multiplicative effects on price.
- If **raw predictive accuracy** is the priority (e.g. an automated valuation model) → the **tuned gradient boosting model** (XGBoost/LightGBM), which captured the lowest test RMSE in this run.
- **Random Forest** is a good middle ground: robust with minimal tuning, natively gives feature importances, and is less sensitive to hyperparameter choices than boosting.
- Whatever model ships to production should be retrained periodically as new sales data becomes available, since housing markets shift (interest rates, local demand) in ways the training window may not capture.