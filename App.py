# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:20:38 2023

@author: Raj Yadav
"""

#import pickle


#import matplotlib.pyplot as plt
#import pandas as pd
#from kneed import KneeLocator
#from sklearn.datasets import make_blobs
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split
#import matplotlib as mtp
#from untitled2 import check_values_greater_than_zero
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics  import r2_score
#from sklearn.tree import DecisionTreeRegressor
#import matplotlib.pyplot as plt   
#import seaborn as sns
#from xgboost import XGBRegressor
import streamlit as st
import os
from streamlit_option_menu import option_menu
from training_class import train
from prediction_class import predict



#st.write(os.chdir(r"C:\Users\Lakshita\Desktop\f1"))
#st.write(os.getcwd())
st.markdown('### Visibility prediction AI Modal ')
with st.sidebar:
    selected = option_menu("Choose an Option", ["Train", 'Predict'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
if selected=="Train":
    
    st.write("Modal Training Wizard")
    if st.button('Start Modal training'):
        train1=train()
        train1.train_data()
        train1.clusterring()
       # train1.folder_to_modal()
        train1.assign_cluster()
        train1.modal_build()
        
else:
    st.write("Enter the data in the fields")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        DRYBULBTEMPF = st.number_input('DRYBULBTEMPF', format="%.2f", step=0.01)
        
    with col2:
        RelativeHumidity = st.number_input('RelativeHumidity', format="%.2f", step=0.01)
        
    with col3:
        WindSpeed = st.number_input('WindSpeed', format="%.2f", step=0.01)
        
    with col1:
        WindDirection = st.number_input('WindDirection', format="%.2f", step=0.01)
        
    with col2:
        SeaLevelPressure = st.number_input('SeaLevelPressure', format="%.2f", step=0.01)
        
    if st.button('Predict'):
        pred=predict(DRYBULBTEMPF,RelativeHumidity,WindSpeed,WindDirection,SeaLevelPressure)
        pred.dataFrameBuilt()
        pred.modal_selection()
        pred.modal_predict()
        
            
