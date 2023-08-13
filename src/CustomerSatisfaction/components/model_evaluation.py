import sys
from abc import ABC, abstractmethod
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from src.CustomerSatisfaction.config.exception import CustomException
from src.CustomerSatisfaction.config.logger import logging

experiment_tracker = Client().active_stack.experiment_tracker


class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE)
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info("Entered the calculate_score method of the MSE class")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"The mean squared error value is: {mse}")
            return mse
        except Exception as e:
            error_message = "Exception occurred in calculate_score method of the MSE class"
            logging.error(error_message)
            raise CustomException(e, sys) from e


class R2Score(Evaluation):
    """
    Evaluation strategy that uses R2 Score
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the R2 score
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values
        Returns:
            R2 score.
        """
        try:
            logging.info(
                "Entered the calculate_score method of the R2Score class")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"The r2 score value is: {r2}")
            return r2
        except Exception as e:
            error_message = "Exception occurred in calculate_score method of the R2Score class"
            logging.error(error_message)
            raise CustomException(e, sys) from e


class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE)
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            logging.info(
                "Entered the calculate_score method of the RMSE class")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"The root mean squared error value is: {rmse}")
            return rmse
        except Exception as e:
            error_message = "Exception occurred in calculate_score method of the RMSE class"
            logging.error(error_message)
            raise CustomException(e, sys) from e


@step(experiment_tracker=experiment_tracker.name)
def evaluation(model: RegressorMixin,
               x_test: pd.DataFrame,
               y_test: pd.Series) -> Tuple[
                   Annotated[float, "r2_score"],
                   Annotated[float, "rmse"]]:
    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        # Calculate the prediction
        prediction = model.predict(x_test)

        # Calculate the MSE, R2 score, and RMSE
        mse = MSE().calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)
        r2_score = R2Score().calculate_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)
        rmse = RMSE().calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        return r2_score, rmse
    except Exception as e:
        error_message = "Exception occurred in evaluation method"
        logging.error(error_message)
        raise CustomException(e, sys) from e
