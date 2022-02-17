
from ml_model.data import get_data
from ml_model.model import get_model
from ml_model.pipeline import get_pipeline
from ml_model.metrics import compute_rmse,compute_mae
from ml_model.mlflow import MLFlowBase
from mlflow.tracking import MlflowClient
import joblib


class Trainer(MLFlowBase):

    def __init__(self):
        
        super().__init__(
            "coastal_app_experiment_trainer")

    def train(self):
        
        self.mlflow_create_run()

        model_name = "keras_v2"


        # get data
        X_train, X_test, y_train, y_test = get_data()

        # log params
        self.mlflow_log_param("model", model_name)

        # create model
        model = get_model(model_name)

        # create pipeline
        pipeline = get_pipeline(model)

        # train
        pipeline.fit(X_train, y_train)

        # make prediction for metrics
        y_pred = pipeline.predict(X_test)

        # evaluate metrics
        score = compute_mae(y_pred, y_test)

        # save the trained model
        joblib.dump(pipeline, "model_best.joblib")

        # push metrics to mlflow
        self.mlflow_log_metric("mae", score)

        # return the gridsearch in order to identify the best estimators and params
        return pipeline
