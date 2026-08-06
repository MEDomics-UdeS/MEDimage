import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.get_institutions_from_ids import get_institutions_from_ids


class CombatNormalization(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible Transformer for ComBat Normalization.
    
    This transformer assumes the input X (DataFrame) contains both the features 
    to be normalized and the column identifying the institution/batch.
    """
    def __init__(
            self,
            institution_col: str = None,
            covariates: list = None,
            drop_institution: bool = True
        ):
        """
        Args:
            institution_col (str): Name of the column in X containing the institution/batch IDs.
                                   If None, tries to derive from Index using util.
            covariates (list): List of column names in X to treat as covariates (biological retention).
            drop_institution (bool): If True, removes the institution column from output.
        """
        self.institution_col = institution_col
        self.covariates = covariates if covariates is not None else []
        self.drop_institution = drop_institution

    def fit(self, X, y=None):
        """
        ComBat calculates parameters on the current batch data provided in transform.
        Standard fit does nothing but validate input exists.
        """
        return self

    def transform(self, X):
        """
        Applies ComBat Normalization.
        """
        # Validate Input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        
        # Avoid modifying the original input
        X_df = X.copy()
        
        # 1. Identify Institutions
        if self.institution_col and self.institution_col in X_df.columns:
            institutions = X_df[self.institution_col]
            # If we plan to drop it later, we don't include it in features matrix
            if self.drop_institution:
                X_df = X_df.drop(columns=[self.institution_col])
        else:
            # Fallback to index-based logic from original code
            institutions = get_institutions_from_ids(pd.Series(X_df.index))

        # Encode institutions to integers (1, 2, 3...) required by logic
        institutions = self._process_institutions(institutions)

        # Check: If < 2 institutions, ComBat fails. Return original.
        if len(np.unique(institutions)) < 2:
            print("Warning: Less than 2 institutions detected. Skipping ComBat.")
            return X_df

        # 2. Prepare Covariates
        # We need to extract covariate data from X if specified
        covars_df = pd.DataFrame({'institution': institutions.flatten()}, index=X_df.index)
        
        for cov in self.covariates:
            if cov in X_df.columns:
                covars_df[cov] = X_df[cov]
                # Remove covariates from the feature matrix to be harmonized?
                # Usually ComBat harmonizes features *adjusting* for covariates. 
                # We keep covariates in X_df usually, but standard combat expects data 
                # to ONLY be the features.
                # Let's separate them:
                X_df = X_df.drop(columns=[cov])
        
        # 3. Handle Single Feature Edge Case (from original code)
        cols_to_restore = []
        if X_df.shape[1] == 1:
            X_df['temp_ones'] = 1
            cols_to_restore.append('temp_ones')

        # 4. Run NeuroCombat
        # neuroCombat expects: data (Features x Samples), covars (Samples x Covars)
        # Note: We transpose X_df because neuroCombat expects Features as Rows
        try:
            results = neuroCombat(
                dat=X_df.T,
                covars=covars_df,
                batch_col='institution'
            )
            
            # 5. Reconstruct DataFrame
            # Result 'data' is Features x Samples
            harmonized_data = pd.DataFrame(results['data'].T, index=X_df.index, columns=X_df.columns)
            
            # Remove temp columns if any
            if cols_to_restore:
                harmonized_data = harmonized_data.drop(columns=cols_to_restore)

            # Re-attach Covariates if they were stripped
            for cov in self.covariates:
                harmonized_data[cov] = covars_df[cov]

            return harmonized_data

        except Exception as e:
            print(f"ComBat failed: {e}. Returning original data.")
            return X

    def _process_institutions(self, institutions):
        """Helper to map institution strings to integers."""
        institutions = pd.Series(institutions)
        unique_inst = institutions.unique()
        mapping = {inst: i+1 for i, inst in enumerate(unique_inst)}
        return institutions.map(mapping).values.reshape(-1, 1)
