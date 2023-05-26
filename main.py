import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
st.title("Loan Default Prediction App")
st.header("XGBoost Model for Loan Default Prediction")
st.subheader("Jesse Russell - May 2023")
st.text("Data Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#")

data = pd.read_csv("defaults_data.csv")

education_mapping = {0:'other'
                    ,1:'graduate school'
                    ,2:'university'
                    ,3:'high school'
                    ,4:'other'
                    ,5:'other'
                    ,6:'other'}

data = data.assign(EDUCATION=data.EDUCATION.map(education_mapping))

sex_mapping = {1:'male'
              ,2:'female'}

data = data.assign(SEX=data.SEX.map(sex_mapping))

marriage_mapping = {0:'other'
                   , 1:'married'
                   , 2:'single'
                   , 3:'other'}

data = data.assign(MARRIAGE=data.MARRIAGE.map(marriage_mapping))


#load label encoder
encoder_m = LabelEncoder()
encoder_m.classes_ = np.load('classes_m.npy',allow_pickle=True)
encoder_e = LabelEncoder()
encoder_e.classes_ = np.load('classes_e.npy',allow_pickle=True)
encoder_s = LabelEncoder()
encoder_s.classes_ = np.load('classes_s.npy',allow_pickle=True)


# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data

st.subheader("Please select relevant features.")
left_column, right_column = st.columns(2)
with left_column:
    inp_gender = st.radio(
        'Gender:',
        np.unique(data['SEX']))

    inp_marriage = st.radio(
        'Marriage:',
        np.unique(data['MARRIAGE']))

    inp_education= st.radio(
        'Education:',
        np.unique(data['EDUCATION']))

input_credit_limit = st.slider('Credit limit', 0, 100000, 1)
input_bill = st.slider('Most recent bill amount', 0, 100000, 1)
input_payment = st.slider('Most recent payment amount', 0,60000, 1)
input_age = st.slider('Age', 0, 100, 1)


if st.button('Make Prediction'):
    #inp_gender = encoder_s.transform(np.expand_dims(inp_gender, -1))
    #inp_marriage = encoder_m.transform(np.expand_dims(inp_marriage, -1))
    #inp_education = encoder_e.transform(np.expand_dims(inp_education, -1))


    # replacing values
    gender_value = 1 if inp_gender == "male" else 2

    marriage_value = 0 if inp_marriage == "other" else (1 if inp_marriage == "married" else
                                                        (2 if inp_marriage == "single" else 0))

    edu_value = 0 if inp_education == "other" else (1 if inp_education == "graduate school" else
                                                        (2 if inp_education == "university" else
                                                         (3 if inp_education == "high school" else 0)))


    inputs = np.expand_dims(
        [int(gender_value), int(marriage_value), int(edu_value),
         input_credit_limit, input_bill, input_payment, input_age], 0)
    prediction = best_xgboost_model.predict_proba(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your loan default prediction is: {np.squeeze(prediction * 100, -1):.2f}%")
