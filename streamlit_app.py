import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Load the dataset and drop the 'Unnamed: 0' column if it exists
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
st.title('Stress Level Predictor')

# Application description
st.write("This application predicts the stress level based on the provided features.")

# Form to input features
st.sidebar.header('Enter the features')

# Fields for the user to input data
def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
    sleep_duration = st.sidebar.number_input('Sleep Duration (hours)', min_value=0, max_value=12, value=0)
    quality_of_sleep = st.sidebar.number_input('Sleep Quality (1 to 10)', min_value=0, max_value=10, value=0)
    physical_activity_level = st.sidebar.number_input('Daily Physical Activity (minutes)', min_value=0, max_value=90, value=0)
    bmi_category = st.sidebar.selectbox('BMI Category', ['Underweight', 'Normal', 'Overweight', 'Obesity'])
    heart_rate = st.sidebar.number_input('Heart Rate (BPM)', min_value=30, max_value=200, value=110)
    daily_steps = st.sidebar.number_input('Daily Steps', min_value=0, max_value=50000, value=8000)
    
    # Convert gender to numeric (0 for Female, 1 for Male)
    gender_numeric = 0 if gender == 'Female' else 1
    
    # Convert BMI category to numeric
    bmi_category_numeric = {
        'Underweight': -1,
        'Normal': 0,
        'Overweight': 1,
        'Obesity': 2
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
    elif 8 < stress_level <= 10:
        return """
        **Very high stress level:** This stress level can have significant negative effects on your health, so it’s essential to take immediate action.
        - Seek professional support if you feel overwhelmed or if stress affects your daily life; a psychologist or therapist can help you develop effective coping strategies.
        - Practice mindfulness exercises to train your mind to focus on the present and reduce the burden of stressful thoughts.
        - Ensure you get enough sleep and improve its quality; consider disconnecting completely from electronic devices an hour before going to bed.
        - Evaluate the possibility of making changes in your environment or routine that may be contributing to your stress level, such as delegating responsibilities or adjusting expectations.
        """

input_df = user_input_features()

# Sidebar message in italics
st.sidebar.markdown("*This project was designed using Kaggle data and provides results for informational purposes only. "
                    "If you are truly experiencing stress-related issues, seek help from a qualified professional.*")

# Ensure the features match exactly those used in training
input_df = input_df[X.columns]

# Prediction
if st.button('Predict'):
    prediction = gbr.predict(input_df)
    st.subheader('Your stress level prediction')
    st.write(f'Based on the data you entered, your level of stress is: {np.round(prediction[0], 2)}.')

    # Display recommendation based on stress level
    recommendation = get_recommendation(prediction[0])
    st.subheader('Potential recommendations')
    st.write(recommendation)
