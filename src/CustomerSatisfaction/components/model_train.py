import sys
from abc import ABC, abstractmethod

import mlflow
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from zenml import step
from zenml.client import Client
import optuna
from src.CustomerSatisfaction.config.exception import CustomException
from src.CustomerSatisfaction.config.logger import logging
from src.CustomerSatisfaction.config.modelConfig import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker


class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model
        Args:
            x_train: training data
            y_train: training labels
        Returns:
            None
        """
        pass
    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameter of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        
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
    
    # For linear regression, there might not be hyperparameter that we want to tune, so we can simply return the score
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.train(x_train, y_train)
        score = reg.score(x_test, y_test)
        return score


class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        try:
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            error_message = "Error occurred from train method from RandomForestModel class"
            logging.error(error_message)
            raise CustomException(e, sys) from e
        
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)
        }
        reg = self.train(x_train, y_train, **params)
        return reg.score(x_test, y_test)


class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = LGBMRegressor(**kwargs)
        try:
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            error_message = "Error occurred from train method from LightGBMModel class"
            logging.error(error_message)
            raise CustomException(e, sys) from e
        
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        parameters = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.99)
        }
        reg = self.train(x_train, y_train, **parameters)
        return reg.score(x_test, y_test)


class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg
    
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 10.0),
        }
        reg = self.train(x_train, y_train, **params)
        return reg.score(x_test, y_test)


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

class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        optimize_func = self.model.optimize
        objective = lambda trial: optimize_func(trial, self.x_train, self.y_train, self.x_test, self.y_test)
        study.optimize(objective, n_trials=n_trials)
        return study.best_trial.params