from ml_model.transformers.function_transformers import SlopeTransformer, DepthToeTransformer,StructureHeightTransformer,RelativeDensityTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import FeatureUnion

def get_pipeline(model):
    
    num_transformer = Pipeline([('scaler', RobustScaler())])
    
    union_custom_transformer = FeatureUnion([
    ('coteng', SlopeTransformer()),
    ('depth_toe', DepthToeTransformer()),
    ('structure_height', StructureHeightTransformer()),
    ('relative_density', RelativeDensityTransformer())
    ])
    
    custom_transformer = Pipeline([('union_custom_transformer',union_custom_transformer), 
          ('scaler_custom',RobustScaler())])
    
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # # Paralellize "num_transformer" and "One hot encoder"
    preprocessor = ColumnTransformer([
        ('num_tr', num_transformer, ['Armour Mass [kg]','Significant Wave Height at Toe [m]','Peak Wave Period [s]','Number of Waves [-]','Notional Permeability [-]']),
        ('cat_tr', cat_transformer, ['Armour Type'])
    ])
    
    preproc = FeatureUnion([
    ('preprocess', preprocessor), 
    ('custom_tr', custom_transformer)
    ])
    
    pipeline = Pipeline([
    ('preprocess',preproc),
    ('model',model)
    ])
    
    return pipeline