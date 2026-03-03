from copy import deepcopy

import numpy as np
import pandas as pd
from pycaret.classification import *
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin

from ..learning.ml_utils import finalize_rad_table, intersect_var_tables


class RandomForestEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            optimization_metric='MCC',
            var_importance_threshold=0.05,
            internal_cv_folds=5,
            optimal_threshold=None,
            use_gpu=False,
            seed=None
        ):
        self.optimization_metric = optimization_metric
        self.var_importance_threshold = var_importance_threshold
        self.internal_cv_folds = internal_cv_folds
        self.optimal_threshold = optimal_threshold
        self.use_gpu = use_gpu
        self.seed = seed
        
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
        # Align tables
        var_table_train, outcome_table_binary_train = intersect_var_tables(var_table_train, outcome_table_binary_train)
        var_table_train = finalize_rad_table(var_table_train)

        # Merge for PyCaret
        temp_data = pd.merge(var_table_train, outcome_table_binary_train, left_index=True, right_index=True)
        target_col = outcome_table_binary_train.columns[0]

        # PyCaret setup
        setup(
            data=temp_data,
            target=target_col,
            feature_selection=True,
            n_features_to_select=1-self.var_importance_threshold,
            fold=self.internal_cv_folds,
            use_gpu=self.use_gpu,
            feature_selection_estimator="rf",
            session_id=self.seed,
            html=False,
            verbose=False
        )

        if self.seed is not None:
            set_config('seed', self.seed)

        # Create RF model. 'balanced' is crucial for your 5/143 imbalance.
        # This penalizes mistakes on the 5 progressors more heavily.
        classifier = create_model('rf', class_weight='balanced', verbose=False)

        # Tune model
        classifier = tune_model(classifier, optimize=self.optimization_metric, verbose=False)
        
        # Assemble dictionary
        model_rf = dict()
        model_rf['algo'] = 'rf'
        model_rf['type'] = 'binary'
        
        # Find threshold
        try:
            model_rf['threshold'] = self.__find_balanced_threshold(classifier, var_table_train, outcome_table_binary_train)
        except Exception as e:
            print(f'Threshold calculation failed: {e}. Defaulting to 0.5')
            model_rf['threshold'] = 0.5
        
        # Store metadata safely
        user_data = var_table_train.Properties.get('userData', {}) if hasattr(var_table_train, 'Properties') else {}
        model_rf['var_info'] = deepcopy(user_data)
        model_rf['var_def'] = deepcopy(user_data.get('variables', {}).get('var_def'))
        model_rf['var_names'] = list(classifier.feature_names_in_)
        model_rf['optimization'] = classifier.get_params()
        
        return model_rf, classifier

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
