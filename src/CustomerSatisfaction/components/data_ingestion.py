import os
import sys
from dataclasses import dataclass

import pandas as pd
from zenml import step

from src.CustomerSatisfaction.config.exception import CustomException
from src.CustomerSatisfaction.config.logger import logging


@dataclass
class DataIngestionConfig:
    """
        Data ingestion Config class which return file paths.
    """
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class IngestData:
    """
        Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        try:
            logging.info(f"Ingesting Data from {self.data_path}")
            return pd.read_csv(self.data_path)
        except Exception as e:
            raise CustomException(e, sys) from e


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Args:
        data_path : Path to the data
    Returns:
        pd.DataFrame : the ingested data
    """
    try:
        ingestion_config = DataIngestionConfig()
        ingest_data = IngestData(data_path=data_path)
        df = ingest_data.get_data()
        os.makedirs(os.path.dirname(
            ingestion_config.raw_data_path), exist_ok=True)
        df.to_csv(ingestion_config.raw_data_path,
                  index=False)  # saving the data
        return df
    except Exception as e:
        raise CustomException(e, sys) from e
