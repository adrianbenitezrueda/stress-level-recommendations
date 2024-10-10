import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Carga del dataset y elimina la columna 'Unnamed: 0' si existe
df = pd.read_csv('stress_level.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Separación de características y objetivo
X = df.drop('Stress Level', axis=1)  # Features
y = df['Stress Level']  # Target

# Entrenamiento del modelo con todos los datos y los mejores parámetros
gbr = GradientBoostingRegressor(
    n_estimators=922,
    max_depth=4,
    learning_rate=0.1090909090909091,
    random_state=42
)
gbr.fit(X, y)

# Título de la aplicación
st.title('Predictor de Nivel de Estrés')

# Descripción de la aplicación
st.write("Esta aplicación predice el nivel de estrés basado en las características proporcionadas.")

# Formulario para introducir características
st.sidebar.header('Introduce las características')

# Campos para que el usuario ingrese datos
def user_input_features():
    gender = st.sidebar.selectbox('Género', ['Femenino', 'Masculino'])
    age = st.sidebar.number_input('Edad', min_value=0, max_value=120, value=30)
    sleep_duration = st.sidebar.number_input('Duración del sueño (horas)', min_value=0, max_value=12, value=0)
    quality_of_sleep = st.sidebar.number_input('Calidad del sueño (1 a 10)', min_value=0, max_value=10, value=0)
    physical_activity_level = st.sidebar.number_input('Actividad física diaria en minutos', min_value=0, max_value=90, value=0)
    bmi_category = st.sidebar.selectbox('Categoría de IMC', ['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad'])
    heart_rate = st.sidebar.number_input('Frecuencia cardiaca (BPM)', min_value=30, max_value=200, value=110)
    daily_steps = st.sidebar.number_input('Pasos diarios', min_value=0, max_value=50000, value=8000)
    
    # Convertir el género a numérico (0 para Femenino, 1 para Masculino)
    gender_numeric = 0 if gender == 'Femenino' else 1
    
    # Convertir la categoría de IMC a numérico
    bmi_category_numeric = {
        'Bajo peso': -1,
        'Normal': 0,
        'Sobrepeso': 1,
        'Obesidad': 2
    }[bmi_category]
    
    data = {
        'Gender': gender_numeric,
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity_level,
        'BMI Category': bmi_category_numeric,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Asegúrate de que las características coincidan exactamente con las usadas en el entrenamiento
input_df = input_df[X.columns]

# Predicción
if st.button('Predecir'):
    prediction = gbr.predict(input_df)
    st.subheader('Predicción de Nivel de Estrés')
    st.write(f'Tu nivel de estrés según tus datos indicados es de: {np.round(prediction[0], 2)}')
