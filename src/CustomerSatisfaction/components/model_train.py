from src.CustomerSatisfaction.config.exception import CustomException
from src.CustomerSatisfaction.config.logger import logging
import sys
from zenml import step
import pandas as pd

@step
def train_model(df:pd.DataFrame)-> None:
    pass