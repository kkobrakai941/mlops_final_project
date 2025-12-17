from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle


import requests
def train_model():
    data = load_iris()
    X = data.data
    y = data.target
    model = LogisticRegression(max_iter=200)
 
    model.fit(X, y)
    with open('/models/model.pkl', 'wb') as f:
        pickle.dump(model, f)


def reload_model_task():
    response = requests.post('http://ml_service:8000/reload_model')
    response.raise_for_status()


with DAG(
    dag_id='model_retrain_dag',
    start_date=datetime(2025, 1, 1),
    schedule_interval='@daily',
    catchup=False
) as dag:
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
    reload_model = PythonOperator(
        task_id='reload_model',
        python_callable=reload_model_task
    )
    train >> reload_model
