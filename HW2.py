import os
import pandas as pd

import mlflow
from mlflow.models import infer_signature

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor


# Фиксируем переменные
experiment_name = 'maxim_grinin'
parent_run_name = 'mkgrinin'


# Чтение данных
def load_and_preprocess_data():
    # Загрузка датасета
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target
    
    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test, housing


# Обучение моделей и логирование с помощью MLflow
def train_and_log_model(experiment_id, model, model_name, X_train, X_test, y_train, y_test, housing):
    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name, nested=True) as child_run:
        # Обучение модели
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Логирование модели
        signature = infer_signature(housing.data, predictions)
        model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)

        # Логирование метрик
        mlflow.evaluate(
            model=model_info.model_uri,
            data=X_test,
            targets=y_test.to_numpy(),
            predictions=predictions,
            model_type='regressor',
            evaluators=['default']
        )
        

# Основная функция
def main():
    # Проверяем, существует ли эксперимент
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is not None::
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    # Начать Parent run с указанием experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=parent_run_name) as parent_run:
        # Чтение и предобработка данных
        X_train, X_test, y_train, y_test, housing = load_and_preprocess_data()

        # Модели
        models = {
            'SGD Regression': SGDRegressor(),
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor()
        }

         # Запуск обучения для каждой модели
        for model_name, model in models.items():
             with mlflow.start_run(experiment_id=experiment_id, run_name=model_name, nested=True) as child_run:
                # Обучение модели
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
        
                # Логирование модели
                signature = infer_signature(housing.data, predictions)
                model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)
        
                # Логирование метрик
                mlflow.evaluate(
                    model=model_info.model_uri,
                    data=X_test,
                    targets=y_test.to_numpy(),
                    predictions=predictions,
                    model_type='regressor',
                    evaluators=['default']
                )

    
if __name__ == '__main__':
    main()