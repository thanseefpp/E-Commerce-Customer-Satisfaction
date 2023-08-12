import sys

import pandas as pd
from zenml import step

from src.CustomerSatisfaction.config.exception import CustomException
from src.CustomerSatisfaction.config.logger import logging
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from abc import ABC,abstractmethod


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
            logging.error("Error occurred from train method from LinearRegressionModel class")
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
            logging.error("Error occurred from train method from RandomForestModel class")
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
            logging.error("Error occurred from train method from LightGBMModel class")
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
            logging.error("Error occurred from train method from XGBoostModel class")
            raise CustomException(e, sys) from e

@step
def train_model(df: pd.DataFrame) -> None:
    pass
