# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_curve, auc, RocCurveDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Interactive deployment: Ask patient info and predict
import streamlit as st

# Load the dataset
data = pd.read_csv('heart1.csv')

# Preprocess the data
data['Sex'] = data['Sex'].map({'M':1, 'F':2})
data['ChestPainType'] = data['ChestPainType'].map({'TA':1, 'ATA':2, 'NAP':3, 'ASY':4})
data['RestingECG'] = data['RestingECG'].map({'Normal':1, 'ST':2, 'LVH':3})
data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y':1, 'N':2})
data['ST_Slope'] = data['ST_Slope'].map({'Up':1, 'Flat':2, 'Down':3})

# Separate features and target variable
x_data = data.drop('HeartDisease', axis=1)
y_data = data['HeartDisease']
# Split data to test and train (train: 80%, test: 20%)
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(x_data, y_data,test_size=0.2,random_state=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(Xtrain)
X_test_scaled = scaler.transform(Xtest)
# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, ytrain)
y_pred_knn = knn.predict(X_test_scaled)
y_pred_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy_knn = accuracy_score(ytest, y_pred_knn)
precision_knn = precision_score(ytest, y_pred_knn)
recall_knn = recall_score(ytest, y_pred_knn)
f1_knn = f1_score(ytest, y_pred_knn)



# Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, ytrain)
y_pred_lr = lr.predict(X_test_scaled)
y_pred_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy_lr = accuracy_score(ytest, y_pred_lr)
precision_lr = precision_score(ytest, y_pred_lr)
recall_lr = recall_score(ytest, y_pred_lr)
f1_lr = f1_score(ytest, y_pred_lr)



def predict_heart_disease(model, scaler, patient_features):
    features_array = np.array(patient_features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    return prediction, probability

st.title("Prediction Heart Disease")
st.write("Enter patient information:")
# Ask for patient info

# 1. Age
age = st.number_input("Patient Age: ",
                      min_value=0, 
                      max_value=120, 
                      step=1)
st.write(f"Patient Age: {age}")

# 2. Sex (gender)
gender = st.radio("Patient Gender: ", ["Male", "Female"],)
sex_map = {"Male": 1, "Female": 2}
sex = sex_map[gender]
st.write(f"Patient Gender as: {gender}")

# 3. User selects chest pain type
cp_map = {
    "Typical Angina (TA)": 1,
    "Atypical Angina (ATA)": 2,
    "Non-Anginal Pain (NAP)": 3,
    "Asymptomatic (ASY)": 4
}
cp_choice = st.radio("Select Chest Pain Type:", list(cp_map.keys()))
chest_pain = cp_map[cp_choice]
st.write(f"Patient chest pain type as: {cp_choice}")

# 4. RestingBP (Resting Blood Pressure)
resting_bp = st.number_input(
    "RestingBP (mm Hg):", 
    min_value=0,   # lower bound
    max_value=250, # upper bound
    step=1
)
st.write(f"Patient Resting Blood Pressure: {resting_bp} mm Hg")

# 5. Cholesterol
cholesterol = st.number_input(
    "Cholesterol (mg/dl):",
    min_value=0,
    max_value=600,
    step=1
)
st.write(f"Patient Cholesterol: {cholesterol} mg/dl")

# 6. FastingBS (Fasting Blood Sugar)
fBS_map = {
    "FastingBS more than or equal to 120": 1,
    "FastingBS less than 120": 0
}
fBS_choice = st.radio("Patient Resting Blood Pressure (RestingBP):", list(fBS_map.keys()))
fasting_bs = fBS_map[fBS_choice]
st.write(f"Patient Resting Blood Pressure as: {fBS_choice}")

# 7. RestingECG (Resting Electrocardiogram Results)
resting_ecg_map = {
    "Normal": 1,
    "ST-T Wave Abnormality (ST)": 2,
    "Left Ventricular Hypertrophy (LVH)": 3
}
ecg_choice = st.radio("RestingECG:", list(resting_ecg_map.keys()))
resting_ecg = resting_ecg_map[ecg_choice]
st.write(f"Patient Resting Electrocardiogram Results: {ecg_choice}")

# 8. MaxHR (Maximum Heart Rate Achieved)
max_hr = st.number_input(
    "MaxHR (beats per minute):", 
    min_value=0,   # approximate lower bound
    max_value=1000,  # approximate upper bound
    step=1
)
st.write(f"Patient Maximum Heart Rate Achieved: {max_hr}")

# 9. ExerciseAngina (Exercise-Induced Angina)
angina_map = {"Yes": 1, "No": 2}
angina_choice = st.radio("Exercise Angina:", ["Yes", "No"])
exercise_angina = angina_map[angina_choice]
st.write(f"Patient with Exercise-Induced Angina: {angina_choice}")

# 10. Oldpeak (ST depression induced by exercise relative to rest)
oldpeak = st.number_input(
    "Oldpeak (ST depression/elevation relative to rest):",
    min_value=-10.0,
    max_value=10.0,
    step=0.1,
    format="%.1f"
)
st.write(f"Patient Oldpeak: {oldpeak}")

# 11. ST_Slope (Slope of the peak exercise ST segment)
st_slope_map = {
    "Up sloping (normal)": 1,
    "Flat (possible ischemia)": 2,
    "Down sloping (abnormal)": 3
}
slope_choice = st.radio("ST Slope:", list(st_slope_map.keys()))
st_slope = st_slope_map[slope_choice]
st.write(f"Patient Slstope of the peak exercise ST segment as: {slope_choice}")


patient_data = [age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]
result, prob = predict_heart_disease(lr, scaler, patient_data)

# Display the result
if result == 1:
    st.error(f"💔 Prediction: Heart Disease")
else:
    st.success(f"💚 Prediction: No Heart Disease")

# Probability display
st.write(f"**Probability of Heart Disease:** {prob * 100:.2f}%")
