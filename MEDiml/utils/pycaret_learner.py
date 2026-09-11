from copy import deepcopy

import numpy as np
import pandas as pd
from pycaret.classification import *
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin

from ..learning.ml_utils import finalize_rad_table, intersect_var_tables


class PyCaretEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            algorithm='xgboost',
            optimization_metric='MCC',
            n_features_to_select=0.05,
            internal_cv_folds=5,
            optimize_threshold=None,
            use_gpu=False,
            seed=None,
            feature_selection_estimator='lightgbm',
            create_model_kwargs=None,
            best_include=None,
            best_exclude=None
        ):
        # Store all parameters as attributes
        self.algorithm = algorithm
        self.optimization_metric = optimization_metric
        self.n_features_to_select = n_features_to_select
        self.internal_cv_folds = internal_cv_folds
        self.optimize_threshold = optimize_threshold
        self.use_gpu = use_gpu
        self.seed = seed
        self.feature_selection_estimator = feature_selection_estimator
        self.create_model_kwargs = create_model_kwargs
        self.best_include = best_include
        self.best_exclude = best_exclude

        # This will hold the "model_info" dictionary result
        self.model_info_ = None
        self.classifier_ = None
        self.selected_features_ = None
        self.selected_features_definitions_ = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Ensure y is a DataFrame for merging in PyCaret logic
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        results, self.classifier_ = self._train_logic(X, y)

        self.model_info_ = results
        self.selected_features_ = results['var_names']
        self.selected_features_definitions_ = results.get('var_def')
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        if self.classifier_ is None:
            raise ValueError("Model has not been fitted yet.")

        probas = self.predict_proba(X)
        threshold = self.model_info_.get('threshold', 0.5)
        return (probas >= threshold).astype(int)

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Filter X to include only features selected during fit
        X_filtered = X[self.selected_features_]
        return self.classifier_.predict_proba(X_filtered)[:, 1]

    def _train_logic(self, var_table_train, outcome_table_binary_train):
        """
        Trains a PyCaret classification model for the given machine learning test.

        Args:
            var_table_train (pd.DataFrame): Radiomics table for the training/learning set.
            outcome_table_binary_train (pd.DataFrame): Outcome table with binary labels for the training/learning set.

        Returns:
            Dict: Dictionary containing info about the trained model.
        """
        # Safety check (make sure that the outcome table and the variable table have the same patients)
        var_table_train, outcome_table_binary_train = intersect_var_tables(var_table_train, outcome_table_binary_train)

        # Finalize the new radiomics table with the remaining variables
        var_table_train = finalize_rad_table(var_table_train)

        # Set up data for PyCaret
        temp_data = pd.merge(var_table_train, outcome_table_binary_train, left_index=True, right_index=True)
        target_col = outcome_table_binary_train.columns[0]

        # PyCaret setup
        setup(
            data=temp_data,
            target=target_col,
            feature_selection=True,
            n_features_to_select=self.n_features_to_select,
            fold=self.internal_cv_folds,
            use_gpu=self.use_gpu,
            feature_selection_estimator=self.feature_selection_estimator,
            session_id=self.seed,
            html=False,
            verbose=False
        )

        from sklearn.metrics import recall_score

        add_metric('sensitivity', 'Sensitivity', recall_score, greater_is_better=True, needs_proba=False)

        # Set seed
        if self.seed is not None:
            set_config('seed', self.seed)

        # Creating the model using PyCaret
        if self.algorithm == 'best':
            classifier = compare_models(
                include=self.best_include,
                exclude=self.best_exclude,
                sort=self.optimization_metric,
                fold=self.internal_cv_folds,
                verbose=False
            )
            resolved_algo = classifier.__class__.__name__
        else:
            classifier = create_model(self.algorithm, **(self.create_model_kwargs or {}), verbose=False)
            resolved_algo = self.algorithm

        # Tuning the model using PyCaret
        classifier = tune_model(classifier, optimize=self.optimization_metric, verbose=False)

        # MEDiml relies on predict_proba for AUC/threshold-based evaluation, but some PyCaret
        # models don't natively support it (e.g. 'svm' -> SGDClassifier with hinge loss, 'ridge'
        # -> RidgeClassifier). Calibrate those so they still produce probability estimates.
        if not hasattr(classifier, 'predict_proba'):
            classifier = calibrate_model(classifier, verbose=False)

        # Saving the information of the model in a dictionary
        model_info = dict()
        model_info['algo'] = resolved_algo
        model_info['type'] = 'binary'

        # Find threshold
        if self.optimize_threshold:
            try:
                model_info['threshold'] = self.__find_balanced_threshold(classifier, var_table_train, outcome_table_binary_train)
            except Exception as e:
                print('Error in finding optimal threshold, it will be set to 0.5:' + str(e))
                model_info['threshold'] = 0.5
        else:
            model_info['threshold'] = 0.5

        user_data = var_table_train.Properties.get('userData', {}) if hasattr(var_table_train, 'Properties') else {}
        model_info['var_info'] = deepcopy(user_data)
        model_info['var_def'] = deepcopy(user_data.get('variables', {}).get('var_def'))
        model_info['var_names'] = list(classifier.feature_names_in_)
        model_info['optimization'] = self.__make_json_safe(classifier.get_params())

        return model_info, classifier

    @staticmethod
    def __make_json_safe(value):
        """Recursively sanitizes get_params() output for JSON serialization.

        Some PyCaret models (e.g. calibrated models, ensembles) nest actual estimator
        objects in their params (e.g. CalibratedClassifierCV's 'estimator' key); those
        aren't JSON-serializable, so they're replaced with their repr() instead of
        crashing the results save step.
        """
        if isinstance(value, dict):
            return {k: PyCaretEstimator.__make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PyCaretEstimator.__make_json_safe(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool, np.generic, np.ndarray)):
            return value
        return repr(value)

    def __find_balanced_threshold(self, model, variable_table, outcome_table_binary) -> float:
        # Align features
        if hasattr(model, 'feature_names_in_'):
            variable_table = variable_table[list(model.feature_names_in_)]

        # Get probabilities
        y_probs = model.predict_proba(variable_table)[:, 1]

        # ROC Calculation
        fpr, tpr, thresholds = metrics.roc_curve(outcome_table_binary.iloc[:, 0], y_probs)

        # Geometric optimization (closest to top-left corner)
        # Distance = sqrt( fpr^2 + (1-tpr)^2 )
        dist = np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2))
        best_idx = np.argmin(dist)

        return thresholds[best_idx]
