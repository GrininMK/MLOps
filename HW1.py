import io
import json
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from typing import Any, Dict, Literal

from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.hooks.S3_hook import S3Hook

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score


# Переменная для хранения бакета на S3
BUCKET = Variable.get("S3_BUCKET")

# Аргументы по умолчанию для всех DAG
DEFAULT_ARGS = {
    "owner" : "Maxim Grinin",
    "email" : "mkgrinin@edu.hse.ru",
    "email_on_failure" : True,
    "email_on_retry" : False,
    "retry": 3,
    "retry_delay" : timedelta(minutes=1)
}

# Список моделей и их имена
model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ])
)

# Функция создания DAG'ов
def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):

    ####### DAG STEPS #######

    # Шаг 1: инициализация пайплайна
    def init(m_name: str, **kwargs) -> Dict[str, Any]:
        start_ts = datetime.now()
        return {
            "model_name": m_name,
            "init_start_ts": start_ts.strftime('%Y-%m-%d %H:%M:%S')
        }

    # Шаг 2: получение данных
    def get_data(m_name: str, **kwargs) -> Dict[str, Any]:
        start_ts = datetime.now()

        # Получаем датасет California housing
        housing = fetch_california_housing(as_frame=True)
        data = pd.concat([housing.data, housing.target], axis=1)

        # Используем boto3 напрямую для загрузки данных на S3
        s3_hook = S3Hook("s3_connection")
        s3_client = s3_hook.get_conn()
        
        filebuffer = io.BytesIO()
        data.to_pickle(filebuffer)
        filebuffer.seek(0)
    
        # Загрузка на S3 с помощью boto3
        s3_client.upload_fileobj(
            Fileobj=filebuffer,
            Bucket=BUCKET,
            Key=f"MaximGrinin/{m_name}/datasets/california_housing.pkl"
        )

        end_ts = datetime.now()
        
        return {
            "get_data_start_ts": start_ts.strftime('%Y-%m-%d %H:%M:%S'),
            "get_data_end_ts": end_ts.strftime('%Y-%m-%d %H:%M:%S'),
            "dataset_size": data.shape
        }
    
    # Шаг 3: подготовка данных
    def prepare_data(m_name: str, **kwargs) -> Dict[str, Any]:
        start_ts = datetime.now()
    
        # Загрузим данные с S3 через boto3 напрямую
        s3_hook = S3Hook("s3_connection")
        s3_client = s3_hook.get_conn()
    
        filebuffer = io.BytesIO()
        s3_client.download_fileobj(
            Bucket=BUCKET,
            Key=f"MaximGrinin/{m_name}/datasets/california_housing.pkl",
            Fileobj=filebuffer
        )
        
        filebuffer.seek(0)
        data = pd.read_pickle(filebuffer)
    
        # Разделение на фичи и таргет
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
        # Стандартизация данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
        # Сохраняем подготовленные данные на S3 через boto3
        for name, dataset in zip(["X_train", "X_test", "y_train", "y_test"], [X_train, X_test, y_train, y_test]):
            filebuffer = io.BytesIO()
            pickle.dump(dataset, filebuffer)
            filebuffer.seek(0)
            s3_client.upload_fileobj(
                Fileobj=filebuffer,
                Bucket=BUCKET,
                Key=f"MaximGrinin/{m_name}/datasets/{name}.pkl"
            )
    
        end_ts = datetime.now()
        return {
            "prepare_data_start_ts": start_ts.strftime('%Y-%m-%d %H:%M:%S'),
            "prepare_data_end_ts": end_ts.strftime('%Y-%m-%d %H:%M:%S'),
            "features": list(X.columns)
        }

    # Шаг 4: обучение модели
    def train_model(m_name: str, **kwargs) -> Dict[str, Any]:
        start_ts = datetime.now()
    
        # Загрузка данных с S3 через boto3 напрямую
        s3_hook = S3Hook("s3_connection")
        s3_client = s3_hook.get_conn()
    
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            filebuffer = io.BytesIO()
            s3_client.download_fileobj(
                Bucket=BUCKET,
                Key=f"MaximGrinin/{m_name}/datasets/{name}.pkl",
                Fileobj=filebuffer
            )
            filebuffer.seek(0)
            data[name] = pd.read_pickle(filebuffer)
    
        # Обучение модели
        model = models[m_name]
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])
    
        # Подсчет метрик
        metrics = {
            "r2_score": r2_score(data["y_test"], prediction),
            "rmse": mean_squared_error(data["y_test"], prediction) ** 0.5,
            "mae": median_absolute_error(data["y_test"], prediction)
        }
    
        end_ts = datetime.now()
        
        return {
            "train_model_start_ts": start_ts.strftime('%Y-%m-%d %H:%M:%S'),
            "train_model_end_ts": end_ts.strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": metrics
        }

    # Шаг 5: сохранение результатов
    def save_results(m_name: str, **kwargs) -> None:
        # Сбор метрик из предыдущих шагов через XCom
        ti = kwargs['ti']
        
        init_metrics = ti.xcom_pull(task_ids='init')
        get_data_metrics = ti.xcom_pull(task_ids='get_data')
        prepare_data_metrics = ti.xcom_pull(task_ids='prepare_data')
        train_model_metrics = ti.xcom_pull(task_ids='train_model')
    
        # Объединяем метрики в один словарь
        result = {
            "init": init_metrics,
            "get_data": get_data_metrics,
            "prepare_data": prepare_data_metrics,
            "train_model": train_model_metrics
        }
    
        # Сохраняем метрики на S3 через boto3
        s3_hook = S3Hook("s3_connection")
        s3_client = s3_hook.get_conn()
        filebuffer = io.BytesIO()
        filebuffer.write(json.dumps(result, indent=4).encode())
        filebuffer.seek(0)
    
        s3_client.upload_fileobj(
            Fileobj=filebuffer,
            Bucket=BUCKET,
            Key=f"MaximGrinin/{m_name}/results/final_metrics.json"
        )

    dag = DAG(
        dag_id=dag_id,
        default_args=DEFAULT_ARGS,
        description=f"DAG for training {m_name}",
        schedule_interval="0 1 * * *",  # ежедневно в 1 час ночи
        start_date=days_ago(1),
        tags=["mlops"]
    )

    # Определяем задачи внутри DAG
    with dag:
        task_init = PythonOperator(
            task_id="init",
            python_callable=init,
            op_kwargs={"m_name": m_name}
        )

        task_get_data = PythonOperator(
            task_id="get_data",
            python_callable=get_data,
            provide_context=True,
            op_kwargs={"m_name": m_name}
        )

        task_prepare_data = PythonOperator(
            task_id="prepare_data",
            python_callable=prepare_data,
            provide_context=True,
            op_kwargs={"m_name": m_name}
        )

        task_train_model = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            provide_context=True,
            op_kwargs={"m_name": m_name}
        )

        task_save_results = PythonOperator(
            task_id="save_results",
            python_callable=save_results,
            provide_context=True,
            op_kwargs={"m_name": m_name}
        )

        # Последовательность выполнения задач
        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results

    return dag

for model_name in models.keys():
    create_dag(f"Maxim_Grinin_{model_name}", model_name)