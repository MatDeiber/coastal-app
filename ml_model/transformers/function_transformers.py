from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

#from sklearn.preprocessing import FunctionTransformer

# def convert_to_coteng(df):
#     return pd.DataFrame(df['Armour Slope [v:h]'].apply(lambda row: float(row.split(':')[1]) / float(row.split(':')[0])))

# def coteng_slope(df):
#     return FunctionTransformer(convert_to_coteng)
    
# def depth_toe(df):
#     return FunctionTransformer(lambda df: pd.DataFrame(df['Water Level [ m Datum]'] - df['Bed Elevation at Toe [m Datum]']))
     
# def structure_height(df):
#     return FunctionTransformer(lambda df: pd.DataFrame(df['Crest Level [m Datum]'] - df['Bed Elevation at Toe [m Datum]']))
    
# def relative_density(df):
#     return FunctionTransformer(lambda df: pd.DataFrame((df['Armour Density [kg/m3]'] - df['Water Density [kg/m3]']) / df['Water Density [kg/m3]']))


class SlopeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        pass

    def transform(self, X, y=None):
        return pd.DataFrame(X['Armour Slope [v:h]'].apply(lambda row: float(row.split(':')[1]) / float(row.split(':')[0])))

    def fit(self, X, y=None):
        return self
    
class DepthToeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        pass

    def transform(self, X, y=None):
        return pd.DataFrame(X['Water Level [ m Datum]'] - X['Bed Elevation at Toe [m Datum]'])

    def fit(self, X, y=None):
        return self
    
class StructureHeightTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        pass

    def transform(self, X, y=None):
        return pd.DataFrame(X['Crest Level [m Datum]'] - X['Bed Elevation at Toe [m Datum]'])

    def fit(self, X, y=None):
        return self
    
class RelativeDensityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        pass

    def transform(self, X, y=None):
        return pd.DataFrame((X['Armour Density [kg/m3]'] - X['Water Density [kg/m3]']) / X['Water Density [kg/m3]'])

    def fit(self, X, y=None):
        return self