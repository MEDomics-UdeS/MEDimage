from sklearn.base import BaseEstimator, ClassifierMixin

from ..utils.rf_learner import RandomForestEstimator
from ..utils.xgboost_learner import XGBoostEstimator


class Estimator(BaseEstimator, ClassifierMixin):
    def __init__(self, algorithm: str, ml_config: dict):
        self.ml_config = ml_config
        self.algorithm = algorithm
        self.estimator_ = None

    def _initialize_estimator(self):
        """Factory algorithm to select the right internal class."""
        if self.algorithm == 'xgboost':
            return XGBoostEstimator(**self.ml_config)
        elif self.algorithm == 'rf':
            return RandomForestEstimator(**self.ml_config)
        else:
            raise ValueError(f"Method {self.algorithm} not supported.")

    def fit(self, X, y):
        self.estimator_ = self._initialize_estimator()
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def save(self, filepath):
        import joblib
        joblib.dump(self, filepath)
