import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

# Load data and preprocess
data = pd.read_csv('survival.csv')
data = data.drop(columns=['Name'])
data = data.dropna()

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])

X = data[['Age', 'Sex', 'BMI', 'Diagnosis', 'Location', 'Resection', 'Infection', 'CT', 'RT', 'Revision']]
y = data['SURV1']

# Random Oversampling
oversample = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = oversample.fit_resample(X, y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_model.fit(X_resampled, y_resampled)

# Streamlit App
def main():
    st.title('Implant Survival Prediction Model (12 months)')
    
    # Sidebar with user inputs
    st.sidebar.header('User Input Features')
    
    # Age
    age_options = ['<18', '18-40', '41-65', '>65']
    age = st.sidebar.selectbox('Age', age_options)

    # BMI
    bmi_options = ['<18.5', '18.5-24.9', '25-29.9', '30-39.9', '>40']
    bmi = st.sidebar.selectbox('BMI', bmi_options)

    # Gender
    sex_options = ['Male', 'Female']
    sex = st.sidebar.selectbox('Gender', sex_options)

    # Diagnosis
    diagnosis_options = ['Primary', 'Metastatic']
    diagnosis = st.sidebar.selectbox('Diagnosis', diagnosis_options)

    # Location
    location_options = ['Upper Extremity', 'Lower Extremity']
    location = st.sidebar.selectbox('Location', location_options)

    # Resection
    resection_options = ['<120', '121-199', '200-299', '>300']
    resection = st.sidebar.selectbox('Resection (mm)', resection_options)

    # Number of surgeries
    revision_options = ['1', '2', '3', '>3']
    revision = st.sidebar.selectbox('Number of surgeries', revision_options)

    # History of infection
    infection_options = ['Yes', 'No']
    infection = st.sidebar.selectbox('History of infection', infection_options)

    # Chemotherapy
    ct_options = ['Yes', 'No']
    ct = st.sidebar.selectbox('Chemotherapy', ct_options)

    # Radiation therapy
    rt_options = ['Yes', 'No']
    rt = st.sidebar.selectbox('Radiation therapy', rt_options)

    # Map user inputs to numerical representations
    age_mapping = {'<18': 0, '18-40': 1, '41-65': 2, '>65': 3}
    bmi_mapping = {'<18.5': 0, '18.5-24.9': 1, '25-29.9': 2, '30-39.9': 3, '>40': 4}
    sex_mapping = {'Male': 1, 'Female': 0}
    diagnosis_mapping = {'Primary': 1, 'Metastatic': 0}
    resection_mapping = {'<120': 0, '121-199': 1, '200-299': 2, '>300': 3}
    revision_mapping = {'1': 0, '2': 1, '3': 2, '>3': 3}
    infection_mapping = {'Yes': 1, 'No': 0}
    ct_mapping = {'Yes': 1, 'No': 0}
    rt_mapping = {'Yes': 1, 'No': 0}

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Age': [age_mapping[age]],
        'Sex': [sex_mapping[sex]],
        'BMI': [bmi_mapping[bmi]],
        'Diagnosis': [diagnosis_mapping[diagnosis]],
        'Location': [location],
        'Resection': [resection_mapping[resection]],
        'Infection': [infection_mapping[infection]],
        'CT': [ct_mapping[ct]],
        'RT': [rt_mapping[rt]],
        'Revision': [revision_mapping[revision]]
    })

    # Make prediction
    y_pred = rf_model.predict(input_data)
    y_proba = rf_model.predict_proba(input_data)[:, 1]

    # Map predicted values to labels
    labels = {0: "Failure", 1: "Survival"}
    prediction_label = labels[y_pred[0]]

    # Display prediction
    st.subheader('Prediction')
    st.write(f'The predicted outcome is {prediction_label}')
    st.write(f'Probability of survival: {y_proba[0]}')

if __name__ == '__main__':
    main()
