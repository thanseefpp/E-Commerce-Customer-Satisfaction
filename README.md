<h3 align="center">E-Commerce Customer Satisfaction</h3>

---

## Dataset is Taken From Kaggle

- [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

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

## 