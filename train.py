import preprocessing
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("project_v1")

def load_data():
    """Load preprocessed data"""
    df = preprocessing.main()    
    return df

def normalise_data(df):
    """Normalize the dataset using column transformations."""
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    ct = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),  
        (OrdinalEncoder(categories='auto'), make_column_selector(dtype_include=object)))  
    X = ct.fit_transform(X)

    return X,y 

def save_model(model, model_filename):
    joblib.dumb(model, model_filename)

def train_model(X,y):
    # Define models and parameters
    models = {
        'Lasso': Lasso,
        'Ridge': Ridge,
        'SVR': SVR,
        'RF': RandomForestRegressor,
        'xgboost': xgb.XGBRegressor
    }

    parameters = {
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'max_iter': [100, 500, 700, 1000, 2000],
            'selection': ['cyclic', 'random']
        },
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10]
        },
        'SVR': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 1, 10],
            'gamma': ['scale', 'auto']
        },
        'RF': {
            'n_estimators': [2, 5, 10, 20, 50, 100, 200],
            'max_depth': [None, 5, 10, 20, 25, 30]
        },
        'xgboost': {
            "max_depth": [2, 3, 4, 5, 6],
            "n_estimators": [500, 600, 700, 750, 800],
            "learning_rate": [0.001, 0.01, 0.1, 1]
        }
    }

    # Iterate over models
    for model_name, model in models.items():
        run_name = f"{model_name}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_name", model_name)

            mlflow.log_param("Model", model_name)

            print(f"Performing GridSearchCV for {model_name}...")
            grid_search = GridSearchCV(estimator=model(), param_grid=parameters[model_name],
                                    cv=LeaveOneOut(), scoring='neg_mean_squared_error', n_jobs=-1, verbose=4,
                                    return_train_score=True)

            # Assuming your features and target are defined globally
            X = preprocessing.transform_features(X)
            y = y

            grid_search.fit(X, y)

            best_params = grid_search.best_params_
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)

            trained_model = model(**grid_search.best_params_)

            mlflow.sklearn.log_model(trained_model, f"{model_name}_model")
            mlflow.log_artifact("mlflow.db", artifact_path="mlflow.db")

            # Make predictions on the entire dataset
            trained_model.fit(X, y)
            y_pred = trained_model.predict(X)

            # Calculate metrics and log
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mlflow.log_metric("RMSE", rmse)

            save_model(trained_model, f"Model/{model_name}_model.pkl")

            print(f"Trained {model_name} model and saved with name {model_name}_model.")

def main():
    df = load_data()
    X, y = normalise_data(df)
    train_model(X, y)