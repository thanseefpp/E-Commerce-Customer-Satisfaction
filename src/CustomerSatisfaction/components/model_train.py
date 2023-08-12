import sys

import pandas as pd
from zenml import step

from src.CustomerSatisfaction.config.exception import CustomException
from src.CustomerSatisfaction.config.logger import logging

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


@step
def train_model(df: pd.DataFrame) -> None:
    pass
