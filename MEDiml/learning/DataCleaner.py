import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.utils import check_random_state


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that cleans features by removing those 
    with too many missing values or too little variation, removes samples with 
    too many missing features, and imputes missing values.
    """
    def __init__(
            self, 
            var_type: str = "continuous",
            imputation: str = "mean",
            missingCutoffpf: float = 0.1,
            missingCutoffps: float = 0.25,
            covCutoff: float = 0.1,
            random_state=None
        ):
        """
        Initializes the DataCleaner with specified parameters for feature and sample filtering and imputation.

        Args:
            var_type (str): Type of variable ("continuous", "hcategorical", "icategorical").
            imputation_method (str): Method of imputation ("mean", "median", "mode", "random").
            missing_cutoff_pf (float): Max % of missing values allowed per feature (column).
            missing_cutoff_ps (float): Max % of missing values allowed per sample (row).
            cov_cutoff (float): Min coefficient of variation allowed per feature.
            random_state (int, RandomState instance or None): Seed for reproducibility.
        
        Returns:
            None
        """
        self.var_type = var_type
        self.imputation_method = imputation
        self.missing_cutoff_pf = missingCutoffpf
        self.missing_cutoff_ps = missingCutoffps
        self.cov_cutoff = covCutoff
        self.random_state = random_state
        
        # Attributes learned during fit
        self.features_to_keep_ = None
        self.imputer_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame=None):
        """
        Learns which features to keep based on missingness and variation thresholds.

        Args:
            X (pd.DataFrame): Input feature data.
            y (pd.DataFrame, optional): Ignored, present for API consistency by convention.
        
        Returns:
            DataCleaner: Returns self.
        """
        # Ensure input is a DataFrame
        X = self._validate_input(X)
        
        # 1. Identify features to keep based on missingness (per feature)
        missing_frac = X.isna().mean()
        features_by_missing = missing_frac[missing_frac <= self.missing_cutoff_pf].index.tolist()

        # 2. Identify features to keep based on Coefficient of Variation (CV)
        # We calculate CV only on the features that passed the missingness check
        X = X[features_by_missing]
        
        # Handle division by zero or near-zero means by adding epsilon
        eps = np.finfo(np.float32).eps
        std = X.std(skipna=True)
        mean = X.mean(skipna=True).abs() + eps
        cv = std / mean
        
        self.features_to_keep_ = cv[cv >= self.cov_cutoff].index.tolist()
        X = X[self.features_to_keep_]

        # 3. Fit the Imputer on the selected features
        self._fit_imputer(X)
        
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies feature selection, sample filtering, and imputation.
        """
        # check is fitted
        if self.features_to_keep_ is None:
            raise RuntimeError("You must fit the transformer before transforming data.")
            
        X = self._validate_input(X)
        
        # 1. Filter Features (Columns)
        # Only keep columns learned during fit
        X_transformed = X[self.features_to_keep_].copy()
        
        # 2. Filter Samples (Rows) based on missingness
        missing_frac_rows = X_transformed.isna().mean(axis=1)
        mask_rows_keep = missing_frac_rows <= self.missing_cutoff_ps
        X_transformed = X_transformed.loc[mask_rows_keep]
        
        # 3. Impute Missing Values
        X_imputed = self._apply_imputation(X_transformed)
        
        # Return as DataFrame to maintain column names
        return pd.DataFrame(X_imputed, columns=self.features_to_keep_, index=X_transformed.index)

    def _fit_imputer(self, X):
        """Helper to initialize and fit the correct imputer logic."""
        # Handle 'random' manually as SimpleImputer doesn't support it
        if "random" in self.imputation_method:
            self.imputer_ = "random" # Marker logic
            return

        # Map methods to SimpleImputer strategies
        strategy_map = {
            "mean": "mean",
            "median": "median",
            "mode": "most_frequent"
        }
        
        # Default logic for icategorical (mode) vs continuous (mean/median)
        if self.imputation_method not in strategy_map:
             # Fallback logic from original class
             if self.var_type == "icategorical":
                 strategy = "most_frequent"
             else:
                 strategy = "mean"
        else:
            strategy = strategy_map[self.imputation_method]

        self.imputer_ = SimpleImputer(strategy=strategy)
        self.imputer_.fit(X)

    def _apply_imputation(self, X):
        """Helper to apply the imputation."""
        if self.imputer_ == "random":
            rng = check_random_state(self.random_state)
            # Custom random imputation logic: fill NaNs with random choice from valid values in that column
            return X.apply(lambda col: col.fillna(
                np.random.choice(col.dropna().values) if not col.dropna().empty else col.mean() # Fallback if empty
            ))
        else:
            return self.imputer_.transform(X)

    def _validate_input(self, X):
        """Ensures X is a DataFrame and handles infinite values."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Replace infs with NaNs (as per original class)
        return X.replace([np.inf, -np.inf], np.nan)
