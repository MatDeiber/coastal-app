from sklearn.model_selection import train_test_split
import pandas as pd

def get_data():
    coastal_data = pd.read_csv('../ml_model/data/database_with_artifical_values.csv', index_col='Id')
    
    X = coastal_data.drop(columns = ['Damage [%]'], axis = 1)
    y = coastal_data['Damage [%]']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    return X_train, X_test, y_train, y_test
     