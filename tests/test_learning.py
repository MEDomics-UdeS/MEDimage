import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from numpyencoder import NumpyEncoder
from sklearn.datasets import make_classification

MODULE_DIR = os.path.dirname(os.path.abspath('./MEDiml/'))
sys.path.append(MODULE_DIR)

from MEDiml.learning.Estimator import Estimator
from MEDiml.utils.rf_learner import RandomForestEstimator
from MEDiml.utils.xgboost_learner import XGBoostEstimator


def _make_radiomics_style_data(n_samples=60, n_features=6, seed=54288):
    """Builds a small synthetic dataset shaped like a MEDiml radiomics table, with the
    `Properties` metadata that PyCaretEstimator._train_logic expects (var_def, userData)."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=4,
        n_redundant=0, random_state=seed
    )
    columns = [f'radVar{i + 1}' for i in range(n_features)]
    index = [f'patient_{i}' for i in range(n_samples)]

    var_def = '||'
    for col in columns:
        var_def += f'{col}:feature_{col}||'

    X_df = pd.DataFrame(X, columns=columns, index=index)
    X_df._metadata += ['Properties']
    X_df.Properties = {
        'userData': {'variables': {'var_def': var_def, 'continuous': np.array(columns)}},
        'VariableNames': columns,
        'Description': 'synthetic'
    }

    y_df = pd.DataFrame({'outcome': y}, index=index)

    return X_df, y_df


def _ml_config(**overrides):
    config = dict(
        optimization_metric='MCC',
        n_features_to_select=0.5,
        internal_cv_folds=3,
        optimize_threshold=False,
        use_gpu=False,
        seed=54288
    )
    config.update(overrides)
    return config


@pytest.mark.parametrize('algorithm', ['lr', 'dt'])
def test_estimator_supports_any_pycaret_algorithm(algorithm):
    X, y = _make_radiomics_style_data()
    estimator = Estimator(algorithm=algorithm, ml_config=_ml_config())
    estimator.fit(X, y)

    proba = estimator.predict_proba(X)
    preds = estimator.predict(X)

    assert proba.shape[0] == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert estimator.estimator_.model_info_['algo'] == algorithm


def test_estimator_best_mode_runs_compare_models():
    X, y = _make_radiomics_style_data()
    estimator = Estimator(
        algorithm='best',
        ml_config=_ml_config(best_include=['lr', 'dt'])
    )
    estimator.fit(X, y)

    proba = estimator.predict_proba(X)
    assert proba.shape[0] == len(X)
    # The winning model's actual class name should be recorded, not the literal "best"
    assert estimator.estimator_.model_info_['algo'] != 'best'
    json.dumps(estimator.estimator_.model_info_['optimization'], cls=NumpyEncoder)


def test_estimator_calibrates_models_without_predict_proba():
    # PyCaret's 'svm' id maps to SGDClassifier(loss='hinge'), which has no predict_proba.
    # PyCaretEstimator must calibrate it so downstream AUC/threshold logic still works.
    X, y = _make_radiomics_style_data()
    estimator = Estimator(algorithm='svm', ml_config=_ml_config())
    estimator.fit(X, y)

    proba = estimator.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert np.all((proba >= 0) & (proba <= 1))

    # Calibrated models nest the raw wrapped estimator in get_params() (e.g. the
    # CalibratedClassifierCV's 'estimator' key); model_info['optimization'] must stay
    # JSON-serializable (as saved via save_json(..., cls=NumpyEncoder) in RadiomicsLearner).
    json.dumps(estimator.estimator_.model_info_['optimization'], cls=NumpyEncoder)


def test_estimator_invalid_algorithm_raises():
    X, y = _make_radiomics_style_data()
    estimator = Estimator(algorithm='not_a_real_model', ml_config=_ml_config())

    with pytest.raises(Exception):
        estimator.fit(X, y)


def test_backward_compatible_wrappers_force_algorithm():
    xgb = XGBoostEstimator(**_ml_config())
    rf = RandomForestEstimator(**_ml_config())

    assert xgb.algorithm == 'xgboost'
    assert rf.algorithm == 'rf'
    assert rf.create_model_kwargs == {'class_weight': 'balanced'}
