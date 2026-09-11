from .pycaret_learner import PyCaretEstimator


class XGBoostEstimator(PyCaretEstimator):
    """Backward-compatible thin wrapper around PyCaretEstimator fixed to the 'xgboost' algorithm."""

    def __init__(self, **kwargs):
        kwargs['algorithm'] = 'xgboost'
        super().__init__(**kwargs)
