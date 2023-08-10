from zenml import pipeline
from src.CustomerSatisfaction.components.data_ingestion import ingest_df
from src.CustomerSatisfaction.components.data_cleaning import clean_data
from src.CustomerSatisfaction.components.model_train import train_model
from src.CustomerSatisfaction.components.model_evaluation import evaluate_model


@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    print(x_train,y_train)
    train_model(df)
    evaluate_model(df)
    