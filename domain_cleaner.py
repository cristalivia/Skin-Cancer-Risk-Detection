import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DomainCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def clean(x):
            if pd.isna(x):
                return np.nan
            if 1 <= x <= 30:
                return x
            elif x == 88:
                return 0
            else:
                return np.nan

        X["_BMI5"] = X["_BMI5"] / 100
        X["_BMI5"] = X["_BMI5"].replace({777: np.nan, 999: np.nan})

        clean_1_cols = [
            'GENHLTH','SMOKE100', 'EXERANY2', 'CVDCRHD4',
            'DIABETE4','HAVARTH4','ADDEPEV3','CHCKDNY2','_ASTHMS1',
            'DIFFWALK','MARITAL', 'CHCSCNC1','CHCOCNC1','_MENT14D','_PHYS14D'
        ]
        X[clean_1_cols] = X[clean_1_cols].replace({7: np.nan, 8: 5, 9: np.nan})

        X['POORHLTH'] = X['POORHLTH'].apply(clean)
        X['EMPLOY1'] = X['EMPLOY1'].replace({9: np.nan})
        
        return X

