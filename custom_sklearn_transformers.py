import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, RFECV

class KNNImputerSubset(KNNImputer):
    
    def __init__(self, n=None, frac=None, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.frac = frac
        self.random_state= random_state
        
    def fit(self, X, y=None):
        X_subset = pd.DataFrame(X).sample(n=self.n,
                                          frac=self.frac, 
                                          random_state=self.random_state,
                                          axis=0)
        
        super().fit(X_subset, y) 
        
        return self
    
    def transform(self, X, y=None):
        return super().transform(X)
    
    
class RFECVSubset(RFECV):
    
    def __init__(self, estimator, n=None, frac=None, random_state=None, **kwargs):
        super().__init__(estimator, **kwargs)
        self.estimator = estimator
        self.n = n
        self.frac = frac
        self.random_state = random_state
    
    def fit(self, X, y=None):
        X_subset = pd.DataFrame(X).sample(n=self.n, frac=self.frac, 
                                          random_state=self.random_state, axis=0)
        
        y_subset = pd.Series(y).sample(n=self.n, frac=self.frac, 
                                       random_state=self.random_state, axis=0)
        
        super().fit(X_subset, y_subset)
        
        return self
    
    def transform(self, X, y=None):
        return super().transform(X)


class SelectivePowerTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, skew_threshold: float, **kwargs):
        self.skew_threshold = skew_threshold
        self.pt = PowerTransformer(**kwargs)
        self.high_skew = None
        self.get_feature_names_out = None
        
    def _reset(self):
        if hasattr(self, "high_skew"):
            del self.high_skew
            del self.get_feature_names_out
    
    def fit(self, X, y=None):
        self._reset()
        
        X_df = pd.DataFrame(X).copy()
        
        self.high_skew = (X_df.skew().abs() >= self.skew_threshold).values
        
        self.pt.fit(X_df.loc[:, self.high_skew])
        
        self.get_feature_names_out = self.get_names(X)
        
        return self
        
    def transform(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        
        X_df.loc[:, self.high_skew] = self.pt.transform(X_df.loc[:, self.high_skew])
        
        return X_df.values
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        
        return self.transform(X)
    
    @staticmethod
    def get_names(X):
        if hasattr(X, "columns"):
            return np.asarray(X.columns, dtype="object")
        else:
            col_num = np.asarray(X).shape[1]
            return np.asarray([f"x{i}" for i in range(col_num)], dtype="object")    
    