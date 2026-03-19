from copy import deepcopy

import numpy as np
import pandas as pd
from pycaret.classification import *
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin

from ..learning.ml_utils import finalize_rad_table, intersect_var_tables


class XGBoostEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            optimization_metric='MCC',
            var_importance_threshold=0.05,
            internal_cv_folds=5,
            optimize_threshold=None,
            use_gpu=False,
            seed=None
        ):
        # Store all parameters as attributes
        self.optimization_metric = optimization_metric
        self.var_importance_threshold = var_importance_threshold
        self.internal_cv_folds = internal_cv_folds
        self.optimize_threshold = optimize_threshold
        self.use_gpu = use_gpu
        self.seed = seed
        
        # This will hold the "model_xgb" dictionary result
        self.model_info_ = None
        self.classifier_ = None
        self.selected_features_ = None

    def fit(self, X, y):
        # 1. Standardize input format (ensure DataFrame)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # 2. Call your existing logic 
        # Note: I am assuming 'intersect_var_tables' and 'finalize_rad_table' 
        # are available in your namespace.
        results, self.classifier_ = self._train_logic(X, y)
        
        # 3. Store results for sklearn
        self.model_info_ = results
        self.selected_features_ = results['var_names']
        self.selected_features_definitions_ = results['var_def']
        self.classes_ = np.unique(y)
        
        return self

    def predict(self, X):
        # Safety check
        if self.selected_features_ is None or self.selected_features_definitions_ is None or self.features_names_in_ is None:
            raise ValueError("Model has no selected features or definitions. " \
            "Ensure that fit() has been called successfully before predict().")
        # Apply the threshold stored in model_info_
        probas = self.predict_proba(X)
        threshold = self.model_info_.get('threshold', 0.5)
        return (probas >= threshold).astype(int)

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Important: Filter X to only include features selected during fit
        features_names = self.selected_features_ or self.selected_features_definitions_ or self.features_names_in_
        X_filtered = X[features_names]
        return self.classifier_.predict_proba(X_filtered)[:, 1]

    def _train_logic(self, var_table_train, outcome_table_binary_train):
        """
        Trains an XGBoost model for the given machine learning test.

        Args:
            var_table_train (pd.DataFrame): Radiomics table for the training/learning set.
            outcome_table_binary_train (pd.DataFrame): Outcome table with binary labels for the training/learning set.
        
        Returns:
            Dict: Dictionary containing info about the trained XGBoost model.
        """
        # Safety check (make sure that the outcome table and the variable table have the same patients)
        var_table_train, outcome_table_binary_train = intersect_var_tables(var_table_train, outcome_table_binary_train)

        # Finalize the new radiomics table with the remaining variables
        var_table_train = finalize_rad_table(var_table_train)

        # Set up data for PyCaret
        temp_data = pd.merge(var_table_train, outcome_table_binary_train, left_index=True, right_index=True)

        # PyCaret setup
        setup(
            data=temp_data,
            feature_selection=True,
            n_features_to_select=1-self.var_importance_threshold,
            fold=self.internal_cv_folds,
            target=temp_data.columns[-1],
            use_gpu=self.use_gpu,
            feature_selection_estimator="xgboost",
            session_id=self.seed
        )

        # Set seed
        if self.seed is not None:
            set_config('seed', self.seed)

        # Creating XGBoost model using PyCaret
        classifier = create_model('xgboost', verbose=False)

        # Tuning XGBoost model using PyCaret
        classifier = tune_model(classifier, optimize=self.optimization_metric)
        
        # Saving the information of the model in a dictionary
        model_xgb = dict()
        model_xgb['algo'] = 'xgb'
        model_xgb['type'] = 'binary'

        # Find threshold
        if self.optimize_threshold:
            try:
                model_xgb['threshold'] = self.__find_balanced_threshold(classifier, var_table_train, outcome_table_binary_train)
            except Exception as e:
                print('Error in finding optimal threshold, it will be set to 0.5:' + str(e))
                model_xgb['threshold'] = 0.5
        else:
            model_xgb['threshold'] = 0.5

        model_xgb['var_info'] = deepcopy(var_table_train.Properties['userData'])
        model_xgb['var_def'] = deepcopy(var_table_train.Properties['userData']['variables']['var_def'])
        model_xgb['var_names'] = list(classifier.feature_names_in_)
        model_xgb['optimization'] = classifier.get_params()
        
        return model_xgb, classifier

    def __find_balanced_threshold(
            self,
            model: object, 
            variable_table: pd.DataFrame, 
            outcome_table_binary: pd.DataFrame
        ) -> float:
        """
        Finds the balanced threshold for the given machine learning test.

        Args:
            model (XGBClassifier): Trained XGBoost classifier for the given machine learning run.
            variable_table (pd.DataFrame): Radiomics table.
            outcome_table_binary (pd.DataFrame): Outcome table with binary labels.
        
        Returns:
            float: Balanced threshold for the given machine learning test.
        """
        # Check is there is a feature mismatch
        if model.feature_names_in_.shape[0] != variable_table.columns.shape[0]:
            variable_table = variable_table.loc[:, model.feature_names_in_]

        # Getting the probability responses for each patient
        patient_ids = list(variable_table.index.values)
        prob_xgb = self.predict(variable_table.loc[patient_ids, :])

        # Calculating the ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(outcome_table_binary.iloc[:, 0], prob_xgb)

        # Calculating the optimal threshold by minizing fpr (false positive rate) and maximizing tpr (true positive rate)
        minimum = np.argmin(np.power(fpr, 2) + np.power(1-tpr, 2))
        
        return thresholds[minimum]