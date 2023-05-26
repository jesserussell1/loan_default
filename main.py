import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
st.header("Loan Default Prediction App")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("defaults_data.csv")
#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

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
input_payment = st.slider('Most recent payment amount', 0,100000, 1)
input_age = st.slider('Age', 0, 100, 1)


if st.button('Make Prediction'):
    inp_gender = encoder.transform(np.expand_dims(inp_gender))
    inp_marriage = encoder.transform(np.expand_dims(inp_marriage))
    inp_education = encoder.transform(np.expand_dims(inp_education))

    inputs = np.expand_dims(
        [int(inp_gender), int(inp_marriage), int(inp_education),
         input_credit_limit, input_bill, input_payment, input_age], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your prediction is: {np.squeeze(prediction, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
