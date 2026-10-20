"""Model definitions. Each function returns an unfitted, full pipeline (preprocessing + model)."""

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
