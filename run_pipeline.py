from src.CustomerSatisfaction.pipeline.train_pipeline import train_pipeline

# from zenml.client import Client


if __name__ == "__main__":
    data_path = "data/olist_customers_dataset.csv"
    # print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path)

# mlflow ui --backend-store-uri "file:/Users/thanseefpp/Library/Application Support/zenml/local_stores/e2b7652e-bb89-46e1-9463-e63dc2b09053/mlruns"
