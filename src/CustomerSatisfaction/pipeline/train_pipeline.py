from zenml import pipeline
from src.CustomerSatisfaction.components.data_ingestion import ingest_df
from src.CustomerSatisfaction.components.data_cleaning import clean_df
from src.CustomerSatisfaction.components.model_train import train_model
from src.CustomerSatisfaction.components.model_evaluation import evaluate_model


@pipeline
def train_pipeline(data_path:str):
    df = ingest_df(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)
    