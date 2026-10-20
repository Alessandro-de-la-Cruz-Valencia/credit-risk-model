"""Model definitions. Each function returns an unfitted, full pipeline (preprocessing + model)."""

from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import RANDOM_STATE
from src.pipeline import build_model_pipeline


def make_logistic(log_transform: bool = True, class_weight: str | None = None) -> Pipeline:
    """Interpretable baseline: scaled logistic regression.

    Scaling puts every coefficient on the same footing (one standard deviation) and helps the
    solver converge. `log_transform` straightens the skewed variables, which a linear model needs.
    """
    model = LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=RANDOM_STATE)
    return build_model_pipeline(model, scale=True, log_transform=log_transform)


def make_lightgbm(class_weight: str | None = None, **params) -> Pipeline:
    """Main model: gradient-boosted trees. No scaling or log needed, trees only use the order of
    each feature's values. `n_jobs=1` because the hyperparameter search parallelizes the folds."""
    model = LGBMClassifier(
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
        **params,
    )
    return build_model_pipeline(model)


# Moderate search space. Small learning rates with a few hundred trees, shallow-ish trees and
# subsampling, because the data has only 10 raw features and overfitting is the main risk.
LGBM_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": randint(200, 800),
    "model__learning_rate": loguniform(0.01, 0.1),
    "model__num_leaves": randint(8, 64),
    "model__min_child_samples": randint(20, 300),
    "model__subsample": uniform(0.6, 0.4),
    "model__subsample_freq": [1],
    "model__colsample_bytree": uniform(0.5, 0.5),
    "model__reg_lambda": loguniform(1e-3, 10),
}
