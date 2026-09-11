from .pycaret_learner import PyCaretEstimator


class RandomForestEstimator(PyCaretEstimator):
    """Backward-compatible thin wrapper around PyCaretEstimator fixed to the 'rf' algorithm.

    Defaults create_model_kwargs to class_weight='balanced', matching the original
    RandomForestEstimator behavior for imbalanced datasets.
    """

    def __init__(self, **kwargs):
        kwargs['algorithm'] = 'rf'
        kwargs.setdefault('create_model_kwargs', {'class_weight': 'balanced'})
        super().__init__(**kwargs)
