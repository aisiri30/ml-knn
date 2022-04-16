# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:37:24 2022

@author: vgbha
"""

import streamlit as st
import pandas as pd
from PIL import Image
import joblib

#Loading our final trained Knn model
model= open("Knn_Classifier.pkl","rb")
knn_clf=joblib.load(model)

st.title("Iris flower species Classification App")
st.sidebar.title("Features")
 #loading files
 
setosa= Image.open('setosa.jpg')
virginica= Image.open('virginica.jpg')
versicolor= Image.open('versicolor.jpg')

#initializing
parameter_list=['Sepal Length (cm)', 'Petal Length (cm)','Sepal Width (cm)','Petal Width (cm)']
parameter_input_values=[]
parameter_default_values=['5.2','3.2','4.2','1.2']
values=[]

#display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
    values= st.sidebar.slider(label=parameter, key=parameter, value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
    parameter_input_values.append(values)
    
input_variables=pd.DataFrame([parameter_input_values], columns=parameter_list, dtype=float)
st.write(input_variables)


prediction = 0
if st.button("Click Here to Classify"):
    prediction = knn_clf.predict(input_variables)

if prediction == 0 :
    st.image(setosa) 
    st.caption('setosa')
elif prediction == 1:
    st.image(versicolor)
    st.caption('versicolor')
else:
    st.image(virginica)
    st.caption('virginica')

 


