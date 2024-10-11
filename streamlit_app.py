import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Dictionary with texts in different languages
texts = {
    'es': {
        'title': 'Predicción de Nivel de Estrés',
        'description': 'Esta aplicación predice el nivel de estrés basado en las características proporcionadas.',
        'sidebar_header': 'Ingresa las características',
        'submit_button': 'Predecir',
        'stress_level': 'Según los datos ingresados, tu nivel de estrés es:',
        'recommendation': 'Recomendaciones potenciales:',
        'sidebar_message': '*Este proyecto fue diseñado usando datos de Kaggle y proporciona resultados solo con fines informativos. Si realmente experimentas problemas relacionados con el estrés, busca ayuda de un profesional cualificado.*'
    },
    'en': {
        'title': 'Stress Level Predictor',
        'description': 'This application predicts the stress level based on the provided features.',
        'sidebar_header': 'Enter the features',
        'submit_button': 'Predict',
        'stress_level': 'Based on the data you entered, your level of stress is:',
        'recommendation': 'Potential recommendations:',
        'sidebar_message': '*This project was designed using Kaggle data and provides results for informational purposes only. If you are truly experiencing stress-related issues, seek help from a qualified professional.*'
    }
}

# Language selection in the sidebar
language = st.sidebar.selectbox('Elige tu idioma / Choose your language:', ('es', 'en'))

# Get texts based on selected language
text = texts[language]

# Load the dataset
df = pd.read_csv('stress_level.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Separate features and target
X = df.drop('Stress Level', axis=1)  # Features
y = df['Stress Level']  # Target

# Train the model with all data and the best parameters
gbr = GradientBoostingRegressor(
    n_estimators=922,
    max_depth=4,
    learning_rate=0.1090909090909091,
    random_state=42
)
gbr.fit(X, y)

# Application title
st.title(text['title'])

# Application description
st.write(text['description'])

# Sidebar header for feature input
st.sidebar.header(text['sidebar_header'])

# Function to collect user input
def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male']) if language == 'en' else st.sidebar.selectbox('Género', ['Femenino', 'Masculino'])
    age = st.sidebar.number_input('Age' if language == 'en' else 'Edad', min_value=0, max_value=120, value=30)
    sleep_duration = st.sidebar.number_input('Sleep Duration (hours)' if language == 'en' else 'Duración del Sueño (horas)', min_value=0, max_value=12, value=0)
    quality_of_sleep = st.sidebar.number_input('Sleep Quality (1 to 10)' if language == 'en' else 'Calidad del Sueño (1 a 10)', min_value=0, max_value=10, value=0)
    physical_activity_level = st.sidebar.number_input('Daily Physical Activity (minutes)' if language == 'en' else 'Actividad Física Diaria (minutos)', min_value=0, max_value=90, value=0)
    bmi_category = st.sidebar.selectbox('BMI Category', ['Underweight', 'Normal', 'Overweight', 'Obesity']) if language == 'en' else st.sidebar.selectbox('Categoría IMC', ['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad'])
    heart_rate = st.sidebar.number_input('Heart Rate (BPM)' if language == 'en' else 'Frecuencia Cardíaca (BPM)', min_value=30, max_value=200, value=110)
    daily_steps = st.sidebar.number_input('Daily Steps' if language == 'en' else 'Pasos Diarios', min_value=0, max_value=50000, value=8000)
    
    gender_numeric = 0 if gender in ['Female', 'Femenino'] else 1
    bmi_category_numeric = {
        'Underweight': -1,
        'Normal': 0,
        'Overweight': 1,
        'Obesity': 2,
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

# Function to get recommendations based on stress level
def get_recommendation(stress_level):
    if language == 'en':
        if stress_level <= 2:
            return """
            **Low stress level:** Excellent! This indicates that you are managing your stress levels well.
            - Continue with your healthy habits such as maintaining a good sleep routine, a balanced diet, and regular exercise.
            - Use this level to strengthen your relaxation practices, such as yoga or meditation, to maintain your well-being.
            - Socialize and connect with friends and family to maintain a positive emotional balance.
            """
        elif 2 < stress_level <= 4:
            return """
            **Moderate-low stress level:** Your stress level is under control, but you could benefit from reinforcing certain areas.
            - Try incorporating active breaks in your work or study day, like stretching or brief walks.
            - Practice deep breathing techniques to relax your mind in moments of tension.
            - Maintain a consistent sleep schedule and avoid screen exposure before bedtime to improve sleep quality.
            """
        elif 4 < stress_level <= 6:
            return """
            **Moderate stress level:** It’s time to pay attention and take steps to manage stress.
            - Consider implementing a daily exercise routine, as physical activity is an excellent way to reduce stress and improve mood.
            - Try meditating for at least 10 minutes a day or incorporate activities you enjoy, such as listening to music, reading, or painting.
            - Set clear boundaries between work and personal life, ensuring you have time to disconnect and rest.
            """
        elif 6 < stress_level <= 8:
            return """
            **High stress level:** It's important to take active steps to reduce your stress level.
            - Dedicate daily time to activities that promote relaxation, such as guided meditation, relaxing baths, or tai chi exercises.
            - Talk to someone you trust about your feelings, as social support can be key to reducing emotional tension.
            - Evaluate your eating habits and make sure to include nutrient-rich foods that benefit your mental health, such as fruits, vegetables, and omega-3-rich fish.
            """
        else:
            return """
            **Very high stress level:** This stress level can have significant negative effects on your health, so it’s essential to take immediate action.
            - Seek professional support if you feel overwhelmed or if stress affects your daily life; a psychologist or therapist can help you develop effective coping strategies.
            - Practice mindfulness exercises to train your mind to focus on the present and reduce the burden of stressful thoughts.
            - Ensure you get enough sleep and improve its quality; consider disconnecting completely from electronic devices an hour before going to bed.
            - Evaluate the possibility of making changes in your environment or routine that may be contributing to your stress level, such as delegating responsibilities or adjusting expectations.
            """
    else:
        if stress_level <= 2:
            return """
            **Nivel de estrés bajo:** ¡Excelente! Esto indica que estás manejando bien tus niveles de estrés.
            - Continúa con tus hábitos saludables, como mantener una buena rutina de sueño, una dieta equilibrada y ejercicio regular.
            - Usa este nivel para fortalecer tus prácticas de relajación, como el yoga o la meditación, para mantener tu bienestar.
            - Socializa y conéctate con amigos y familiares para mantener un equilibrio emocional positivo.
            """
        elif 2 < stress_level <= 4:
            return """
            **Nivel de estrés moderado-bajo:** Tu nivel de estrés está bajo control, pero podrías beneficiarte de reforzar ciertas áreas.
            - Intenta incorporar pausas activas en tu día de trabajo o estudio, como estiramientos o caminatas breves.
            - Practica técnicas de respiración profunda para relajar tu mente en momentos de tensión.
            - Mantén un horario de sueño consistente y evita la exposición a pantallas antes de dormir para mejorar la calidad del sueño.
            """
        elif 4 < stress_level <= 6:
            return """
            **Nivel de estrés moderado:** Es momento de prestar atención y tomar medidas para manejar el estrés.
            - Considera implementar una rutina de ejercicio diario, ya que la actividad física es una excelente manera de reducir el estrés y mejorar el estado de ánimo.
            - Intenta meditar al menos 10 minutos al día o incorpora actividades que disfrutes, como escuchar música, leer o pintar.
            - Establece límites claros entre el trabajo y la vida personal, asegurándote de tener tiempo para desconectar y descansar.
            """
        elif 6 < stress_level <= 8:
            return """
            **Nivel de estrés alto:** Es importante tomar medidas activas para reducir tu nivel de estrés.
            - Dedica tiempo diario a actividades que promuevan la relajación, como meditación guiada, baños relajantes o ejercicios de tai chi.
            - Habla con alguien en quien confíes sobre tus sentimientos, ya que el apoyo social puede ser clave para reducir la tensión emocional.
            - Evalúa tus hábitos alimenticios y asegúrate de incluir alimentos ricos en nutrientes que beneficien tu salud mental, como frutas, verduras y pescados ricos en omega-3.
            """
        else:
            return """
            **Nivel de estrés muy alto:** Este nivel de estrés puede tener efectos negativos significativos en tu salud, así que es esencial tomar medidas inmediatas.
            - Busca apoyo profesional si te sientes abrumado o si el estrés afecta tu vida diaria; un psicólogo o terapeuta puede ayudarte a desarrollar estrategias efectivas de afrontamiento.
            - Practica ejercicios de atención plena para entrenar tu mente a enfocarse en el presente y reducir la carga de pensamientos estresantes.
            - Asegúrate de dormir lo suficiente y mejorar su calidad; considera desconectarte completamente de los dispositivos electrónicos una hora antes de acostarte.
            - Evalúa la posibilidad de hacer cambios en tu entorno o rutina que puedan estar contribuyendo a tu nivel de estrés, como delegar responsabilidades o ajustar expectativas.
            """

input_df = user_input_features()
st.sidebar.markdown(text['sidebar_message'])

# Ensure the features match exactly those used in training
input_df = input_df[X.columns]

# Prediction
if st.button(text['submit_button']):
    prediction = gbr.predict(input_df)
    st.subheader(text['stress_level'])
    st.write(f'{np.round(prediction[0], 2)}')

    # Display recommendation based on stress level
    recommendation = get_recommendation(prediction[0])
    st.subheader(text['recommendation'])
    st.write(recommendation)
