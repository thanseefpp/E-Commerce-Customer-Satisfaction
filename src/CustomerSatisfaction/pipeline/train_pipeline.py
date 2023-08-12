from zenml import pipeline

from src.CustomerSatisfaction.components.data_cleaning import clean_data
from src.CustomerSatisfaction.components.data_ingestion import ingest_data
from src.CustomerSatisfaction.components.model_evaluation import evaluation
from src.CustomerSatisfaction.components.model_train import train_model


@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model, x_test, y_test)
