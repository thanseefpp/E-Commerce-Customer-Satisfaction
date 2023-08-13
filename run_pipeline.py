from src.CustomerSatisfaction.pipeline.train_pipeline import train_pipeline
from src.CustomerSatisfaction.components.data_cleaning import clean_data
from src.CustomerSatisfaction.components.data_ingestion import ingest_data
from src.CustomerSatisfaction.components.model_evaluation import evaluation
from src.CustomerSatisfaction.components.model_train import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    training = train_pipeline(
        ingest_data(),
        clean_data(),
        train_model(),
        evaluation(),
    )

    training.run()
    
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

