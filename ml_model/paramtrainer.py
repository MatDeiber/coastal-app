from ml_model.data import get_data
from ml_model.model import get_model
from ml_model.pipeline import get_pipeline
from ml_model.mlflow import MLFlowBase
from sklearn.model_selection import GridSearchCV
from ml_model.metrics import compute_rmse

import joblib

class ParamTrainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            "coastal_app_experiment")
        
        # self.EXPERIMENT_NAME = "coastal_app_experiment"
        # self.client = MlflowClient()
        
        # try:
        #     self.experiment_id = self.client.create_experiment(self.EXPERIMENT_NAME)
        # except BaseException:
        #     self.experiment_id = self.client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id
        
    def train(self, params):

        # results
        models = {}

        # iterate on models
        for model_name, model_params in params.items():
            
            print(f'Processing {model_name}')
            
            self.mlflow_create_run()

            hyper_params = model_params["hyper_params"]

            # log params
            for key, value in hyper_params.items():
                self.mlflow_log_param(key, value)


            # get data
            X_train, X_test, y_train, y_test = get_data()

            # log params
            self.mlflow_log_param("model", model_name)

            # create model
            model = get_model(model_name)

            # create pipeline
            pipeline = get_pipeline(model)
            

            # create gridsearch object
            grid_search = GridSearchCV(
                pipeline,
                param_grid=hyper_params,
                cv=5,
                verbose = 1,
                scoring = 'neg_mean_absolute_error')

            
            # train with gridsearch
            grid_search.fit(X_train, y_train)
            self.mlflow_log_param('best_params', grid_search.best_params_)

            # score gridsearch
            score = grid_search.score(X_test, y_test)

            # save the trained model
            joblib.dump(pipeline, f"{model_name}.joblib")

            # push metrics to mlflow
            self.mlflow_log_metric("score", score)

            # return the gridsearch in order to identify the best estimators and params
            models[model_name] = grid_search

        return models