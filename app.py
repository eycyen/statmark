import streamlit as st
import pandas as pd
import joblib

# Load the production model (XGBoost) and the scaler
xgb_model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load the dataset to get the mean values for background features
df = pd.read_csv('data/finalscore_cleaned.csv')

# --- MAIN PAGE HEADER ---
st.title("COM2502 Project: Predicting Final Scores")
st.markdown("This application predicts student final scores based on an XGBoost machine learning model. Adjust the most impactful features in the sidebar to see how they affect the final grade.")

# --- SIDEBAR: TOP FEATURES INPUT ---
st.sidebar.header("Student Profile")
st.sidebar.markdown("Adjust the top features that affect the score:")

# Using the most important features from the XGBoost Feature Importance chart
tutoring = st.sidebar.slider("Tutoring Sessions Per Week", 0, 10, 1)
hours_studied = st.sidebar.slider("Hours Studied", 0, 15, 5)
exam_anxiety = st.sidebar.slider("Exam Anxiety Score", 0.0, 10.0, 5.0)
stress_level = st.sidebar.slider("Stress Level", 0, 10, 5)
previous_gpa = st.sidebar.number_input("Previous GPA", 0.0, 4.0, 2.5, step=0.1)
diet_quality_text = st.sidebar.selectbox("Diet Quality", ["Poor", "Average", "Good"])
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)

# Map the categorical text input back to numerical value
diet_mapping = {"Poor": 0, "Average": 1, "Good": 2}
diet_quality = diet_mapping[diet_quality_text]

# --- BACKGROUND FEATURES (MEANS) ---
# Calculate default mean values for less important features to keep the UI clean
age = df["Age"].mean()
sleep_hours = df["Sleep_Hours"].mean()
screen_time = df["Screen_Time"].mean()
part_time_job = df["Part_Time_Job"].mean()
internet_quality = df["Internet_Quality"].mean()
extracurricular = df["Extracurricular"].mean()
family_income_level = df["Family_Income_Level"].mean()
gender_male = df["Gender_Male"].mean()
gender_non_binary = df["Gender_Non-Binary"].mean()
study_method_offline = df["Study_Method_Offline"].mean()
study_method_online = df["Study_Method_Online"].mean()

# --- DATAFRAME CREATION ---
input_data = pd.DataFrame({
    "Age": [age],
    "Hours_Studied": [hours_studied],
    "Attendance": [attendance],
    "Sleep_Hours": [sleep_hours],
    "Stress_Level": [stress_level],
    "Screen_Time": [screen_time],
    "Previous_GPA": [previous_gpa],
    "Part_Time_Job": [part_time_job],
    "Diet_Quality": [diet_quality],
    "Internet_Quality": [internet_quality],
    "Extracurricular": [extracurricular],
    "Tutoring_Sessions_Per_Week": [tutoring],
    "Family_Income_Level": [family_income_level],
    "Exam_Anxiety_Score": [exam_anxiety],
    "Gender_Male": [gender_male],
    "Gender_Non-Binary": [gender_non_binary],
    "Study_Method_Offline": [study_method_offline],
    "Study_Method_Online": [study_method_online]
})

# --- PREDICTION LOGIC ---
if st.button("Predict Final Score", type="primary"):
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make the prediction
    prediction = xgb_model.predict(scaled_input)
    final_score = float(prediction[0])
    
    # Display the result
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if final_score >= 60:
        st.success("Great job! The predicted score indicates a passing grade.")
    else:
        st.warning("Warning: The predicted score is low. Consider increasing study hours or tutoring.")
        
    st.metric(label="Predicted Final Score", value=f"{final_score:.2f}")