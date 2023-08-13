from src.CustomerSatisfaction.config.exception import CustomException
from src.CustomerSatisfaction.config.logger import logging
import sys
import pandas as pd
from src.CustomerSatisfaction.components.data_cleaning import DataCleaning,DataPreprocessStrategy

def get_data_for_test():
    try:
        df = pd.read_csv("../../../data/olist_customers_dataset.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["review_score"], axis=1, inplace=True)
        return  df.to_json(orient="split")
    except Exception as e:
        error_message = "Error occurred from get_data_for_test method"
        logging.error(error_message)
        raise CustomException(e, sys) from e