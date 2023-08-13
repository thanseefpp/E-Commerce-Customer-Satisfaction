import sys
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.CustomerSatisfaction.config.modelConfig import ModelNameConfig
from src.CustomerSatisfaction.config.exception import CustomException
from src.CustomerSatisfaction.config.logger import logging
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from abc import ABC,abstractmethod
import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series)-> None:
        """
        Train the model
        Args:
            x_train: training data
            y_train: training labels
        Returns:
            None
        """
        pass


class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            error_message = "Error occurred from train method from LinearRegressionModel class"
            logging.error(error_message)
            raise CustomException(e, sys) from e

class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = RandomForestRegressor(**kwargs)
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            error_message = "Error occurred from train method from RandomForestModel class"
            logging.error(error_message)
            raise CustomException(e, sys) from e

class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LGBMRegressor(**kwargs)
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            error_message = "Error occurred from train method from LightGBMModel class"
            logging.error(error_message)
            raise CustomException(e, sys) from e
    
class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = xgb.XGBRegressor(**kwargs)
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            error_message = "Error occurred from train method in XGBoostModel class"
            logging.error(error_message)
            raise CustomException(e, sys) from e

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model = None

        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")
        return model.train(x_train, y_train)
    except Exception as e:
        error_message = "Error occurred from train_model method"
        logging.error(error_message)
        raise CustomException(e, sys) from e
