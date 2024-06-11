# ====================================
# Heart disease prediction 
# ====================================

# Import required libraries
import numpy as np
import pandas as pd
import streamlit as st
import time
import pickle
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy.stats.mstats import winsorize

st.set_page_config(page_title="Heart Disease Predictor App", layout="wide")

# Introduction
st.write("""
        # Heart Disease Predictor App
        """)
st.write("""
         
## This app predicts symptoms of heart disease

The dataset for this prediction was obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UC Irvine ML repository . 
""")

# Heart diseases illustration
image = Image.open("heart-disease-predict.JPG")
st.image(image, width=700)

# Define custom functions and classes
# Outlier handling
def winsorize_with_pandas(series, limits):
    return winsorize(series, limits=limits)

class handling_outliers(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self  
    
    def transform(self, X, y=None):
        heart_data = X.copy()
        cols_to_winsorize = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        heart_data[cols_to_winsorize] = heart_data[cols_to_winsorize].apply(winsorize_with_pandas, limits=[0.01, 0.01])
        return heart_data

# Adding some dummy data to fit the pipeline initially
dummy_data = pd.DataFrame({
    'age': [50, 60],
    'trestbps': [130, 140],
    'chol': [250, 240],
    'thalach': [150, 160],
    'oldpeak': [2.3, 3.5],
    'sex': [1, 0],
    'cp': [1, 2],
    'fbs': [0, 1],
    'restecg': [0, 1],
    'exang': [0, 1],
    'slope': [1, 2],
    'ca': [0, 1],
    'thal': [2, 3]
})

# Column Transformer
transformer = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first', categories=[
        [0, 1],  # sex
        [1, 2, 3, 4],  # cp
        [0, 1],  # fbs
        [0, 1, 2],  # restecg
        [0, 1],  # exang
        [0, 1, 2],  # slope
        [0, 1, 2, 3],  # ca
        [1, 2, 3]  # thal
    ]), ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']),
], remainder='passthrough')

# Scaling
scaler = MinMaxScaler()

# Pipeline
pipeline = Pipeline([
    ('outlier', handling_outliers()),
    ('transformer', transformer),
    ('scaler', scaler)
])

# Fit the pipeline with dummy data
pipeline.fit(dummy_data)

# Collect user input features into a dataframe
st.sidebar.header('User Input Features:')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

def user_input_features():
    st.sidebar.header('Manual Input')
    cp = st.sidebar.slider('Chest pain type', 1, 4, 2)
    wcp = ["Typical angina", "Atypical angina", "Non angina", "Asymptomatic"][cp - 1]
    st.sidebar.write("Type of Chest pain:", wcp)
    
    thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
    slope = st.sidebar.slider("Slope of the peak exercise ST segment", 0, 2, 1)
    oldpeak = st.sidebar.slider("ST depression induced", 0.0, 6.2, 1.0)
    exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
    ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
    thal = st.sidebar.slider("Result of thallium test", 1, 3, 1)
    
    restecg = st.sidebar.slider('Result of resting electrocardiographic', 0, 2, 1)
    wrestecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][restecg]
    st.sidebar.write("Result of resting electrocardiographic:", wrestecg)
    
    chol = st.sidebar.slider("Serum cholesterol level in the blood (mg/dl)", 126, 564, 180)
    trestbps = st.sidebar.slider("Blood pressure at rest (mmHg)", 94, 200, 120)
    fbs = st.sidebar.selectbox("Fasting blood sugar level > 120 mg/dl?", ['No', 'Yes'])
    fbs = 1 if fbs == 'Yes' else 0
    
    sex = st.sidebar.selectbox("Sex of patients", ['Female', 'Male'])
    sex = 0 if sex == 'Female' else 1
    
    age = st.sidebar.slider("Age of patients", 29, 77, 30)
    
    data = {
        'cp': cp,
        'thalach': thalach,
        'slope': slope,
        'oldpeak': oldpeak,
        'exang': exang,
        'ca': ca,
        'thal': thal,
        'restecg': restecg,
        'chol': chol,
        'trestbps': trestbps,
        'fbs': fbs,
        'sex': sex,
        'age': age
    }
    return pd.DataFrame(data, index=[0])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    input_df = user_input_features()

# Display input dataframe
st.write("User Input Features:")
st.write(input_df)# List of image file names
image_files = ['man-heart-attack.JPG', 'woman-heart-attack.JPG']
desired_width = 180
desired_height = 180

col1, col2, col3, col4 = st.columns(4)
for idx, image_file in enumerate(image_files):
    img = Image.open(image_file)
    resized_img = img.resize((desired_width, desired_height))
    if idx == 0:
        col1.image(resized_img, caption=image_file, use_column_width=True)
    else:
        col2.image(resized_img, caption=image_file, use_column_width=True)

# Loading images
heartdisease = Image.open('heart-disease.JPG')
strongheart = Image.open('strong-heart.JPG')

# Load the best model
loaded_model = None
with open("knn_for_heart_disease_prediction.pkl", "rb") as f:
    loaded_model = pickle.load(f)

if st.sidebar.button('Predict'):
    st.write("Processing...")
    if loaded_model is not None:
        try:
            # Apply preprocessing pipeline to input data
            processed_df = pipeline.transform(input_df)
            st.write("Processed data:")
            st.write(processed_df)
            # Predict using the loaded model
            prediction = loaded_model.predict(processed_df)
            result = 'You have some symptoms of heart disease. Please consult with a doctor.' if prediction[0] else 'You do not have symptoms of heart disease. You can verify it to a doctor'

            st.subheader('Prediction Result:')
            with st.spinner('Wait for it...'):
                time.sleep(4)

            st.success(f"Prediction of this app is: {result}")
            st.image(heartdisease if prediction[0] else strongheart)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Model not loaded properly.")



    




