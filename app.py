import streamlit as st
import pandas as pd
from io import StringIO
import logging
from inference import predict, load_model
from preprocessing import inference_preprocessor

# # Configure the logger
logging.basicConfig(level=logging.INFO)
# Create a logger instance
logger = logging.getLogger("MyConsoleLogger")
# Create a StreamHandler and add it to the logger
# stream handler will send your logs to the terminal
# We call the terminal standard output professionally
# You can send logs to a file, for that you create, you guessed it, a filehandler
console_handler = logging.StreamHandler()
# Create a formatter for your stream handler
# basically sets the format the log output should look like
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(console_handler)

model_path = 'artifacts/model/RF_model.joblib'
preprocessor_path = 'artifacts/preprocessor/preprocessor.joblib'

# @st.cache_resource
@st.experimental_memo
def load_models(model_path, preprocessor_path):  
    model = load_model(model_path)
    preprocessor = load_model(preprocessor_path)
    return model, preprocessor

def main():
    st.title('Dystonia Improvement Score Prediction') 
    st.markdown('Predicting the dystonia improvement scores given DBS programming parameters and electrode localisation')
    st.sidebar.markdown("## Variable Selector")
    space = st.sidebar.selectbox("Atlas space",
                                 ("ACPC", "MNI"),
                                 index = 0#None,
                                #  placeholder = "Select Atlas space"
                                ) ### upgrade streamlit version... why so old

    # file uploader: upload excel file for inference
    uploaded_file = st.file_uploader("Choose a file", type = {"xlsx"})
    if uploaded_file is not None:
        # if type== "xlsx":
            
        # elif type == "csv":
        #     df = pd.read_csv(uploaded_file)
        df = pd.read_excel(uploaded_file)
        

        model, preprocessor = load_models(model_path, preprocessor_path)
        preprocessed_data = inference_preprocessor(df, space, preprocessor)
        predictions = []
        if st.button('Predict dystonia improvement score'):
            prediction = model.predict(preprocessed_data)
            df['Predicted score'] = prediction
        st.write(df)
        


def run(model, preprocessor, data,space):
    # prediction = None
    # put the prediction in a try catch block to handle errors safely
    try:
        preprocessed_data = inference_preprocessor(data, space, preprocessor)
        prediction = predict(model, preprocessed_data)

        # predictions = []
        # if st.button('Predict dystonia improvement score'):
            
        #     for index, row in df.iterrows():
        #         prediction = run(model, preprocessor, row, space)
        #         predictions.append(prediction)
        #     df['Predicted score'] = predictions

        #     st.write(df)
    except Exception as e:
        logger.error(f"An error occurred: {e} ")

    return prediction


if __name__ == "__main__":
    main()