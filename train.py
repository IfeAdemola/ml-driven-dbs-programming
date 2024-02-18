import preprocessing
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import argparse
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import GridSearchCV,LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("project_18_02_2024")

def load_data(path):
    """
    Reads an Excel file from the specified path and returns its contents as a DataFrame.
    Args:
        path (str): The path to the Excel file that needs to be read.
    Returns:
        pandas.DataFrame: A DataFrame containing the data read from the Excel file.
    """
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{path}' was not found.")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing the Excel file '{path}': {e}")
    return df


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

            file_path = f"artifacts/model/{model_name}_model.joblib"
            save_model(trained_model, file_path)
        
    return trained_model

def save_model(model,file_path):
    """
    Save a machine learning model to a file using joblib.

    Args:
        model: The trained machine learning model to be saved.
        file_path (str): The file path where the model will be saved.
    """
    try:
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error while saving the model: {str(e)}")

def main(data_path, to_delete, space, target):
    df = load_data(data_path)
    output = preprocessing.normalise_data(df, to_delete, space, target)
    model = train_model(*output['data'])
    save_model(output['preprocessor'], "artifacts/preprocessor.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Process data from a specified path")
    parser.add_argument("--data_path", type=str, help="Path to the raw data file")
    parser.add_argument("--target", type=str, help="Specify the target variable: old total or new total")
    parser.add_argument("--space", type=str, help="Specify the space: MNI or ACPC")
    parser.add_argument('--to_delete', nargs='+', help="List of columns to be deleted")

    args = parser.parse_args()
    main(args.data_path, args.to_delete, args.space, args.target)
