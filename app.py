import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import time
import pickle
import warnings
warnings.filterwarnings ("ignore")
from absenteeism_module import*

st.title("Mubarake steel of Isfahan")
image = Image.open('OIG1.XJsRLfhRLXuEo5FeoXFo (1).jpg')
st.image(image, use_column_width=True)

def Input_Output():
    data = st.file_uploader("upload file", type={"csv", "txt"})
    if data is not None:
        df = pd.read_csv (data)
        model = absenteeism_model ('model', 'scaler')
    
        model.load_and_clean_data('Absenteeism_new_data.csv')
    
    result = ""

    if st.button("Click here to Predict"):
        result = model.predicted_outputs()
        with st.spinner('Wait for it...'):
            time.sleep(5)
    st.info ('The output is as follows: ')
    st.write(result)
    
if __name__ == '__main__':
    Input_Output ()


