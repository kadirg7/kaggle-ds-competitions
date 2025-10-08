import streamlit as st
import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'random_forest_model.joblib')
COLUMNS_PATH = os.path.join(BASE_DIR, 'model_columns.joblib')



@st.cache_resource
def load_model():
   
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_data
def load_columns():
  
    columns = joblib.load(COLUMNS_PATH)
    return columns


model = load_model()
model_columns = load_columns()



st.title('Bank Customer Subscription Prediction App')
st.write('This app predicts the probability of a client subscribing to a bank term deposit.')


st.header('Enter Client Information')

age = st.number_input('Age', min_value=18, max_value=100, value=40)
balance = st.number_input('Average Yearly Balance (€)', value=1500)
duration = st.number_input('Last Contact Duration (seconds)', value=250)
campaign = st.number_input('Number of Contacts During this Campaign', min_value=1, value=2)
pdays = st.number_input('Days Since Previous Campaign Contact (-1: first contact)', value=-1)
previous = st.number_input('Number of Contacts Before this Campaign', value=0)


job_options = ['management', 'technician', 'blue-collar', 'admin.', 'services', 'retired', 'self-employed', 'entrepreneur', 'unemployed', 'housemaid', 'student', 'unknown']
job = st.selectbox('Job', options=job_options)

marital_options = ['married', 'single', 'divorced']
marital = st.selectbox('Marital Status', options=marital_options)

education_options = ['secondary', 'tertiary', 'primary', 'unknown']
education = st.selectbox('Education Level', options=education_options)

default = st.selectbox('Has Credit in Default?', options=['no', 'yes'])
housing = st.selectbox('Has Housing Loan?', options=['no', 'yes'])
loan = st.selectbox('Has Personal Loan?', options=['no', 'yes'])
contact = st.selectbox('Contact Communication Type', options=['cellular', 'unknown', 'telephone'])
poutcome = st.selectbox('Previous Campaign Outcome', options=['unknown', 'failure', 'other', 'success'])


if st.button('Predict Subscription Probability'):

    user_data = {
        'age': age,
        'balance': balance,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'poutcome': poutcome,
        'day': 15, 
        'month': 'may' 
    }
    input_df = pd.DataFrame([user_data])

    
    input_dummies = pd.get_dummies(input_df, drop_first=True)

    
    final_input = input_dummies.reindex(columns=model_columns, fill_value=0)

    
    probability = model.predict_proba(final_input)[:, 1]
    
   
    st.subheader('Prediction Result')
    st.write(f'The probability of this client subscribing is: **{probability[0]*100:.2f}%**')

    if probability[0] > 0.5:
        st.success('This client is a potential subscriber!')
    else:
        st.error('The probability of this client subscribing is low.')
