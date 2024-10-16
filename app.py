import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def data():
    df = pd.read_csv("Predict Student Dropout and Academic Success.csv")
    df.Target=df.Target.map({"Graduate":1,"Dropout":2,"Enrolled":3})
    return df

def add_prediction(input_data):
    
    df=data()
    with open("model.pkl","rb") as pickle_in:
        classifier=pickle.load(pickle_in)
    
    with open("scaler.pkl","rb") as scaler_in:
        scaler=pickle.load(scaler_in)
        
    input_array=np.array(list(input_data.values())).reshape(1,-1)
    input_scaled=scaler.transform(input_array)
    prediction=classifier.predict(input_scaled)
    
    st.subheader("Prediction")
    st.write("The student is:") 
    if prediction ==1:
        st.write("Graduate")
    elif prediction ==2:
        st.write("Dropout")
    else:
        st.write("Enrolled")


def add_sidebar():
    df=data()
    st.header("Students Attributes")
    slider_label=[
        ("Course","Course"),
        ("Previous qualification (grade)","Previous qualification (grade)"),
        ("Father's occupation","Father's occupation"),
        ("Admission grade","Admission grade"),
        ("Age at enrollment","Age at enrollment"),
        ("Curricular units 1st sem (approved)","Curricular units 1st sem (approved)"),
        ("Curricular units 1st sem (grade)","Curricular units 1st sem (grade)"),
        ("Curricular units 2nd sem (evaluations)","Curricular units 2nd sem (evaluations)"),
        ("Curricular units 2nd sem (approved)","Curricular units 2nd sem (approved)"),
        ("Curricular units 2nd sem (grade)","Curricular units 2nd sem (grade)")
        ]
    input_dict = {}
    for label, key in slider_label:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0,
            max_value=int(df[key].max())
        )
    return input_dict



def main():
    st.set_page_config(
        page_title="Student Acedemic Success prediction App",
        layout="wide",
        initial_sidebar_state="expanded")
    input_data= add_sidebar()

    with st.container():
        st.title("Student Academic Success Prediction")
        st.write("This app is designed for the prediction of Student Success based on the provided attributes.")
        
    if st.button("Predict"):
        add_prediction(input_data)
if __name__=="__main__":
    main()