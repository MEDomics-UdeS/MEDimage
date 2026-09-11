Learning
--------

This section walks you through setting up the consolidated master configuration file (``config.yaml``) for the machine learning pipeline. 
Instead of multiple JSON files in the previous versions, all parameters are now managed in a single YAML structure, 
allowing for easier maintenance and the use of YAML anchors for consistency (e.g., shared seeds).

The configuration is separated into the following subdivisions:

* :ref:`Study Metadata`
* :ref:`Experiment Design Parameters`
* :ref:`Variables Definition`
* :ref:`Data Cleaning Parameters`
* :ref:`Data Normalization Parameters`
* :ref:`Feature Set Reduction Parameters`
* :ref:`Machine Learning Parameters`

Study Metadata
^^^^^^^^^^^^^^

Defines high-level experiment identifiers and global variables.

.. code-block:: yaml

   study_metadata:
     var_study: "var1"
     combinations: ["var1"]
     seed: &global_seed 54288  # YAML anchor used to sync seeds across the pipeline

Experiment Design Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used to define the data splitting and validation strategy. You can define multiple profiles and select the active one.

.. code-block:: yaml

   design:
     active_method: "CrossValidation" # Options: "Random", "CrossValidation", "Bootstrapping"
     
     Random:
       method: "SubSampling"
       nSplits: 10
       stratifyInstitutions: 1
       testProportion: 0.33
       seed: *global_seed

     CrossValidation:
       method: "StratifiedKFold"
       nFolds: 5
       nRepeats: 10
       seed: *global_seed

     Bootstrapping:
       method: "Out-of-Bag"
       nIterations: 1000
       seed: *global_seed

Variables Definition
^^^^^^^^^^^^^^^^^^^^

Defines the data sources and maps them to specific cleaning and reduction profiles defined later in the file.

.. code-block:: yaml

   variables:
     var1:
       nameType: "RadiomicsFull"
       path: "setToFeaturesinWorkspace"
       scans: ["CECT"]
       rois: ["tumor"]
       imSpaces: ["image"]
       cleaning_profile: "default"
       normalization: "combat"
       reduction_method: "FDA"

Data Cleaning Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

Defines how missing values and low-variance features are handled.

.. code-block:: yaml

   data_cleaning:
     default:
       continuous:
         missingCutoffps: 0.25 # Max % missing features per sample
         covCutoff: 0.1        # Min coefficient of variation
         missingCutoffpf: 0.1  # Max % missing samples per feature
         imputation: "mean"    # Options: "mean", "median", "random"

Data Normalization Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Aims to remove batch effects (e.g., multicenter differences).

.. code-block:: yaml

   normalization:
     standardCombat: "RUN"
     standardization:
       perClass: 0
       perInstitution: 0
     minmax:
       min: 0
       max: 1

Feature Set Reduction Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters for reducing high-dimensional feature sets (like Radiomics) to a stable subset.

.. code-block:: yaml

   feature_reduction:
     FDA:
       nSplits: 100
       corrType: "Spearman" # Options: "Spearman", "Pearson"
       threshStableStart: 0.5
       threshInterCorr: 0.7
       minNfeatStable: 100
       minNfeat: 10
       seed: *global_seed

Machine Learning Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defines the algorithm and hyperparameter optimization settings.

.. code-block:: yaml

   modeling:
     method: "xgboost" # Any PyCaret classification model ID ("xgboost", "rf", "lr", "dt", "lightgbm",
                        # "catboost", ...), or "best" to auto-select via PyCaret's compare_models()
     optimization_metric: "MCC"
     cv_folds: 5
     n_features_to_select: 0.5 # If < 1, interpreted as a fraction of features to keep; if >= 1, interpreted as the number of features to keep.
     feature_selection_estimator: "lightgbm" # Optional. Model used by PyCaret for setup()'s feature selection step.
     create_model_kwargs: {} # Optional. Extra kwargs forwarded to PyCaret's create_model(), e.g. {class_weight: "balanced"}
     best_include: null # Optional. Restricts compare_models() to this list of model IDs when method is "best"
     best_exclude: null # Optional. Excludes these model IDs from compare_models() when method is "best"

.. note::
   For rare-event studies (e.g., only 5 positive cases), it is highly recommended to use a ``Random Forest`` (or any model) with ``create_model_kwargs: {class_weight: "balanced"}``.

Full Configuration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is a complete example of a ``config.yaml`` file incorporating all the sections discussed above. You can copy this into your project workspace to get started.

.. code-block:: yaml

   # ==============================================================================
   # Master Configuration for Machine Learning Pipeline
   # ==============================================================================
   # Note: All lines below are indented by 3 spaces to satisfy the readthedocs directive

   study_metadata:
     var_study: "var1"
     combinations: ["var1"]
     seed: &global_seed 54288

   design:
     active_method: "CrossValidation"
     
     Random:
       method: "SubSampling"
       nSplits: 10
       stratifyInstitutions: 1
       testProportion: 0.33
       seed: *global_seed

     CrossValidation:
       method: "StratifiedKFold"
       nFolds: 5
       nRepeats: 10
       seed: *global_seed

     Bootstrapping:
       method: "Out-of-Bag"
       nIterations: 1000
       seed: *global_seed

   variables:
     var1:
       nameType: "RadiomicsFull"
       path: "path/to/features/workspace"
       scans: ["CECT"]
       rois: ["tumor"]
       imSpaces: ["image"]
       cleaning_profile: "default"
       normalization: "combat"
       reduction_method: "FDA"

   data_cleaning:
     default:
       continuous:
         missingCutoffps: 0.25
         covCutoff: 0.1
         missingCutoffpf: 0.1
         imputation: "mean"

   normalization:
     standardCombat: "RUN"
     standardization:
       perClass: 0
       perInstitution: 0
     minmax:
       min: 0
       max: 1

   feature_reduction:
     FDA:
       nSplits: 100
       corrType: "Spearman"
       threshStableStart: 0.5
       threshInterCorr: 0.7
       minNfeatStable: 100
       minNfeat: 10
       seed: *global_seed

   modeling:
     method: "xgboost"
     optimization_metric: "MCC"
     cv_folds: 5
     n_features_to_select: 0.5