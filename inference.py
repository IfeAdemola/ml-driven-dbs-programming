import pandas as pd
import numpy as np
import joblib
import argparse
# from preprocessing import inference_preprocessor

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
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument("--ct_path", type=str, help="Path to the transformer file")
    
    parser.add_argument("--gender", type=str, help="Target column name")
    parser.add_argument("--dystonia type", type=str, help="Target column name")
    parser.add_argument("--amplitude", type=str, help="Target column name")
    parser.add_argument("--voltage", type=str, help="Target column name")
    parser.add_argument("--pulsewidth", type=str, help="Target column name")
    parser.add_argument("--frequency", type=str, help="Target column name")
    parser.add_argument("--position", type=str, help="Target column name")
    parser.add_argument("--transformation_matrix", type=str, help="Target column name")
    parser.add_argument("--leadtype", type=str, help="Target column name")
    
    parser.add_argument("--data_path", type=str, help="Path to the data file")
    args = parser.parse_args()

    model = load_model(args.model_path)
    data = pd.read_csv(args.data_path)
    data = inference_preprocessor(data)
    prediction = predict(model, data)
    print(f"Prediction: {prediction}")