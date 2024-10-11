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

# Función para obtener recomendaciones basadas en el nivel de estrés
def get_recommendation(stress_level):
    if stress_level <= 2:
        return """
        **Nivel de estrés bajo:** ¡Excelente! Esto indica que estás manejando bien tus niveles de estrés. 
        - Continúa con tus hábitos saludables como mantener una buena rutina de sueño, una alimentación balanceada y ejercicio regular.
        - Aprovecha este nivel para fortalecer tus prácticas de relajación, como yoga o meditación, para mantener tu bienestar.
        - Socializa y conecta con amigos y familiares para mantener un equilibrio emocional positivo.
        """
    elif 2 < stress_level <= 4:
        return """
        **Nivel de estrés moderado-bajo:** Tu nivel de estrés está bajo control, pero podrías beneficiarte de reforzar ciertas áreas.
        - Intenta incorporar pausas activas en tu jornada laboral o de estudio, como estiramientos o breves caminatas.
        - Practica técnicas de respiración profunda para relajar la mente en momentos de tensión.
        - Mantén un horario de sueño consistente y evita la exposición a pantallas antes de dormir para mejorar la calidad del sueño.
        """
    elif 4 < stress_level <= 6:
        return """
        **Nivel de estrés moderado:** Es momento de prestar atención y tomar medidas para gestionar el estrés.
        - Considera implementar una rutina diaria de ejercicio, ya que la actividad física es una excelente forma de reducir el estrés y mejorar el ánimo.
        - Prueba meditar al menos 10 minutos al día o incorpora actividades que disfrutes, como escuchar música, leer, o pintar.
        - Establece límites claros entre tu vida laboral y personal, asegurándote de tener tiempo para desconectar y descansar.
        """
    elif 6 < stress_level <= 8:
        return """
        **Nivel de estrés alto:** Es importante tomar acción activa para reducir tu nivel de estrés.
        - Dedica tiempo diario a actividades que promuevan la relajación, como la meditación guiada, baños relajantes, o ejercicios de tai chi.
        - Habla con alguien de confianza sobre tus sentimientos, ya que el apoyo social puede ser clave para reducir la tensión emocional.
        - Evalúa tus hábitos alimenticios y asegúrate de incluir alimentos ricos en nutrientes que beneficien tu salud mental, como frutas, verduras, y pescados ricos en omega-3.
        """
    elif 8 < stress_level <= 10:
        return """
        **Nivel de estrés muy alto:** Este nivel de estrés puede tener efectos negativos significativos en tu salud, por lo que es fundamental tomar medidas inmediatas.
        - Busca apoyo profesional si sientes que el estrés es abrumador o afecta tu vida cotidiana; un psicólogo o terapeuta puede ayudarte a desarrollar estrategias efectivas de afrontamiento.
        - Practica ejercicios de mindfulness para entrenar tu mente a enfocarse en el presente y reducir la carga de pensamientos estresantes.
        - Asegúrate de dormir suficiente y mejorar la calidad de tu sueño, y considera desconectar completamente de dispositivos electrónicos una hora antes de ir a la cama.
        - Evalúa la posibilidad de realizar cambios en tu entorno o en tu rutina que puedan estar contribuyendo a tu nivel de estrés, como delegar responsabilidades o ajustar expectativas.
        """

input_df = user_input_features()

# Mensaje en cursiva en la barra lateral
st.sidebar.markdown("*Este proyecto fue diseñado mediante el uso de datos de Kaggle y da resultados meramente informativos. "
                    "Si realmente tienes algún problema con el estrés, acude a un profesional cualificado.*")

# Asegúrate de que las características coincidan exactamente con las usadas en el entrenamiento
input_df = input_df[X.columns]

# Predicción
if st.button('Predecir'):
    prediction = gbr.predict(input_df)
    st.subheader('Predicción de Nivel de Estrés')
    st.write(f'Tu nivel de estrés según tus datos indicados es de: {np.round(prediction[0], 2)}')

    # Mostrar la recomendación basada en el nivel de estrés
    recommendation = get_recommendation(prediction[0])
    st.subheader('Recomendaciones')
    st.write(recommendation)
