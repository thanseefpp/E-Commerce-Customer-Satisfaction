<h3 align="center">E-Commerce Customer Satisfaction</h3>

---

![training_and_deployment_pipeline_updated](https://github.com/thanseefpp/E-Commerce-Customer-Satisfaction/assets/62167887/fce9441f-a847-4319-a6e7-52e1ed133e0c)

---

**Problem statement**: For a given customer's historical data, we are tasked to predict the review score for the next order or purchase. We will be using the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing charges from various dimensions: from order status, price, payment, freight performance to customer location, product attributes and finally, reviews written by customers. The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. In order to achieve this in a real-world scenario, we will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.

## 1 - Installation

```
pip install -r requirements.txt
```
## 2 - Setup ZenML

```
zenml init
```

## To See ZenML Dashboard
```
zenml up
```
visit the ULR Given in the bash

## 3 - Integrating MLFlow With ZenML
```
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_tracker --flavor=mlflow
zenml stack register local_with_mlflow -m default -a default -o default -d mlflow
```

## Incase If You Get an Error like this follow the below code to upgrade mlflow

attributeerror: module 'sklearn.metrics' has no attribute 'scorers'

```
pip install --upgrade mlflow
```

### 4 - Running The Application

```
streamlit run streamlit_app.py
```