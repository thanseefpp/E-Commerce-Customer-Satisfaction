import sys

import pandas as pd
from zenml import step

from src.CustomerSatisfaction.config.exception import CustomException
from src.CustomerSatisfaction.config.logger import logging


@step
def train_model(df: pd.DataFrame) -> None:
    pass
