"""Machine learning model training and inference wrappers."""

from typing import Any


class ModelPlaceholder:
    """Temporary placeholder for the ML model wrapper."""

    def fit(self, X: Any, y: Any) -> None:
        return None

    def predict(self, X: Any) -> Any:
        return None