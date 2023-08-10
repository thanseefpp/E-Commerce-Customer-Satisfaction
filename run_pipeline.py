from src.CustomerSatisfaction.pipeline.train_pipeline import train_pipeline

if __name__ == "__main__":
    data_path = "data/olist_customers_dataset.csv"
    train_pipeline(data_path)
