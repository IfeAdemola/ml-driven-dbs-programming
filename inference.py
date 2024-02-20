import pandas as pd
import numpy as np
import joblib
import argparse
from preprocessing import inference_preprocessor

def predict(model, data):
    """
    Predict the class of a given data point.

    Args:
        model (object): The trained model object.
        data (numpy array): The data point to be classified.

    Returns:
        int: The predicted class.
    """
    prediction = model.predict(data)
    prediction = float(prediction)
    return prediction

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


def load_model(file_path):
    """
    Load a machine learning model from a file using joblib.

    Args:
        file_path (str): The file path from which to load the model.

    Returns:
        model: The loaded machine learning model.
    """
    try:
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        print(f"Error while loading the model: {str(e)}")
        return None
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Process data")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--ct_path", type=str, help="Path to the preprocessor transformer file")
    parser.add_argument("--data_path", type=str, help="Path to the data file for inference")
    # parser.add_argument("--target", type=str, help="Specify the target variable: old totakl or new total")
    parser.add_argument("--space", type=str, help="Specify the space: MNI or ACPC")

    args = parser.parse_args()
    model = load_model(args.model_path)
    preprocessor = load_model(args.ct_path)
    data_path = args.data_path
    # target = float(args.target)
    space = args.space

    df = load_data(data_path)
    data = inference_preprocessor(df, space, preprocessor)

    prediction = predict(model, data)
    print(f"Prediction: {prediction}")