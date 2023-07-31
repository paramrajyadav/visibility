# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:20:43 2023

@author: Raj Yadav
"""
import pickle
#import shutil
#import matplotlib.pyplot as plt
import pandas as pd
#from kneed import KneeLocator
#from sklearn.datasets import make_blobs
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split
#import matplotlib as mtp
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics  import r2_score
#from sklearn.tree import DecisionTreeRegressor
#import matplotlib.pyplot as plt   
#import seaborn as sns
#from xgboost import XGBRegressor
import streamlit as st
import os
#from streamlit_option_menu import option_menu


class predict():
    
    def __init__(self,v1,v2,v3,v4,v5):
        self.v1=float(v1)
        self.v2=float(v2)
        self.v3=float(v3)
        self.v4=float(v4)
        self.v5=float(v5)
        self.df=[]
        self.y=int
        self.Y=[]
    def dataFrameBuilt(self):
        data1 = {
    'DRYBULBTEMPF': [self.v1],
    'RelativeHumidity': [self.v2],
    'WindSpeed': [self.v3],
    'WindDirection': [self.v4],
    'SeaLevelPressure': [self.v5]}


# Creating the DataFrame
        self.df = pd.DataFrame(data1)
        #st.write(self.df)
        
    
    def modal_selection(self):
        
        #folder_path = os.getcwd() + "\Kmeans"
        
        #file_path = os.path.join(folder_path, "cluster_modal.pkl")
        #st.write(file_path)
        #st.write(r"C:\Users\Lakshita\Desktop\f1\kmeans\cluster_modal.pkl")
        file = open("kmeans.pkl", 'rb')
        kmeans=pickle.load(file)
        self.y=kmeans.predict(self.df)
        self.y = int(self.y)
        #st.write(self.y)
        #st.write(type(self.y))
        
        file.close()

    def modal_predict(self):
        #folder_path = os.getcwd() + "\standard scaler for cluster "+str(self.y)
        #st.write(folder_path)
    
        #file_path = "standard scaler for cluster "+str(i)+".pkl"
        #st.write(file_path)
        #st.write(r"C:\Users\Lakshita\Desktop\f1\standard scaler for cluster 1\cluster_modal.pkl")
        file = open("standard scaler for cluster "+str(self.y)+".pkl", 'rb')
        sc=pickle.load(file)
        file.close()
        self.df2=sc.transform(self.df)
        
        
        
        #folder_path = os.getcwd() + "\modal for cluster "+str(self.y)
        
        
        #file_path = os.path.join(folder_path, "modal.pkl")
        #st.write(file_path)
        #st.write(r"C:\Users\Lakshita\Desktop\f1\modal for cluster 1\modal.pkl")
        with open("modal for cluster "+str(self.y)+".pkl", 'rb') as file:
            #content = fil"C:\Users\Lakshita\Desktop\f1\modal for cluster 1\modal.pkl"e.read()
            #st.write(content)

            ml=pickle.load(file)
            self.Y=ml.predict(self.df2)
        
        
        st.write(self.Y)

        
