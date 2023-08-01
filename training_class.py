# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:20:43 2023

@author: Raj Yadav
"""
import pickle
import shutil
#import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator
#from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#import matplotlib as mtp
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import r2_score
from sklearn.tree import DecisionTreeRegressor
#import matplotlib.pyplot as plt   
#import seaborn as sns
from xgboost import XGBRegressor
import streamlit as st
import os
#from streamlit_option_menu import option_menu


class train():
    
    def __init__(self):
        self.X=[]
        self.s=[]
        self.z=[]
        self.Y=[]
        self.X_train=[]
        self.X_test=[]
        self.y_train=[]
        self.y_test=[]
        self.error=[]
        self.model={}
        self.error=[]
        self.randomForest_error=[]
        self.xgb_error=[]
        self.dt_error=[]
        self.dt_model = DecisionTreeRegressor()
        self.xgb1 = XGBRegressor()
        self.clf = RandomForestRegressor()
    def train_data(self):
        #st.write(os.getcwd())
        ds=pd.read_csv("Visibility_data.csv")
        ds=ds.drop(['DATE','Precip','WETBULBTEMPF','DewPointTempF','StationPressure'],axis=1)
        self.X = ds.drop(['VISIBILITY'],axis=1)
        y = ds['VISIBILITY']

        # Assuming you have a pandas DataFrame called 'ds' with numerical columns
        
        # Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = self.X.quantile(0.25)
        Q3 = self.X.quantile(0.75)
        
        # Calculate the IQR (Interquartile Range)
        IQR = Q3 - Q1
        
        # Define a multiplier to adjust the range for outlier detection (e.g., 1.5 or 3)
        multiplier = 1.5
        
        # Create a boolean mask to identify the outliers
        outlier_mask = (self.X < (Q1 - multiplier * IQR)) | (self.X > (Q3 + multiplier * IQR))
        
        # Remove the outliers from the DataFrame
        clean_ds = self.X[~outlier_mask.any(axis=1)]
        
        # The resulting 'clean_ds' DataFrame will have the outliers removed
        self.X=clean_ds  
        #scaler = StandardScaler()

        #X_scaled = scaler.fit_transform(self.X)
        column_names=ds.columns.values
        self.Y=ds["VISIBILITY"]
        self.z=self.Y
        self.X=ds.drop(["VISIBILITY"],axis=1)
        #st.write(self.X)
        #st.write(self.Y)

        #st.write(self.z)        #return self.X
        
        
    def clusterring(self):
        wcss=[]
        for i in range (1,11):
          kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) # initializing the KMeans object
          kmeans.fit(self.X) # fitting the data to the KMeans Algorithm
          wcss.append(kmeans.inertia_)
          
          
        
        
        
        kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
        self.s=kn.knee
        #st.write(self.s)
        return self.s
        
     
    def folder_to_modal(self):
        #st.write("2")
        original_path=os.getcwd()
        #st.write("11")
        for i in range(self.s):
            #st.write(i)
            folder_path=os.getcwd() + "\modal for cluster " + str(i)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
# If the folder doesn't exist, create it
                os.makedirs(folder_path)
            else:
                os.mkdir(folder_path)
    #st.write(folder_path)
        
                
    def assign_cluster(self):
        kmeans = KMeans(n_clusters=self.s, init='k-means++', random_state= 42)  
        y_predict= kmeans.fit_predict(self.X)
        
        
        
        #folder_path = os.getcwd() + "\kmeans"  # Use forward slash (/) for file paths in Python
        #st.write(folder_path)
        #st.write(self.s)
        #st.write(str(self.s))
        
        file_path = "kmeans.pkl"
        #st.write(file_path)

    # Pickle the model and save it to the file
        
        
        if os.path.exists(file_path):
            os.remove(file_path)
       # with open(file_path, 'wb') as file:
        #    pickle.dump(kmeans, file)
        
        #file_path = os.path.join(folder_path, "cluster_modal.pkl")
        file = open(file_path, 'wb')
        pickle.dump(kmeans, file)
        file.close()
        
        
        
        cluster=y_predict
        self.z=pd.DataFrame(self.z)
        cluster = pd.DataFrame(cluster)
        self.z["Cluster"]=cluster
        print(type(cluster))
        print(type(self.z))
        self.X["Cluster"]=cluster
        column_names=self.X.columns.values
        #st.write(self.X)



    def randomforest_(self):
        #st.write("3")
        #clf=RandomForestRegressor()
        param_grid = {"n_estimators": [10, 50, 100]
                    , "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                                   "max_depth": range(2, 3, 1), "max_features": ['sqrt','log2']}
        #clf=RandomForestRegressor()
        grid = GridSearchCV(estimator=self.clf, param_grid=param_grid, cv=5,  verbose=3)
        grid.fit(self.X_train, self.y_train)
    
        #extracting the best parameters
        criterion = grid.best_params_['criterion']
        max_depth = grid.best_params_['max_depth']
        max_features = grid.best_params_['max_features']
        n_estimators = grid.best_params_['n_estimators']
    
        #creating a new model with the best parameters
    
        clf1 = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion,
                                        max_depth=max_depth, max_features=max_features)
        # training the mew model
        clf1.fit(self.X_train, self.y_train)
        predict_clf=clf1.predict(self.X_test)
       # self.randomForest_error
        self.randomForest_error = r2_score(self.y_test,predict_clf)
        return clf1



    def xg_boost_(self):
        #st.write("33")
        param_grid_xgboost = {
    
        'learning_rate': [0.5, 0.1, 0.01, 0.001]
        ,'max_depth': [3, 5, 10, 20]
        ,'n_estimators': [10, 50, 100, 200]
        }
        xgb1 = XGBRegressor()
        # Creating an object of the Grid Search class
        grid= GridSearchCV(#XGBRegressor(objective='reg:squarederror'),
            xgb1,
                           param_grid_xgboost, verbose=3,cv=5)
        # finding the best parameters
        grid.fit(self.X_train, self.y_train)
    
        # extracting the best parameters
        learning_rate = grid.best_params_['learning_rate']
        max_depth = grid.best_params_['max_depth']
        n_estimators = grid.best_params_['n_estimators']
        print(learning_rate,max_depth,n_estimators)
        print(grid.best_params_)
    
      # creating a new model with the best parameters
        xgb =XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        verbosity=1)#,
        #objective='reg:squarederror')
      # training the mew model
        xgb.fit(self.X_train, self.y_train)
        predict_xgb=xgb.predict(self.X_test)
        
        self.xgb_error = r2_score(self.y_test,predict_xgb)
        return xgb




    def decession_tree_(self):
        #st.write("333")
        param_grid_decisionTree = {'criterion': ['squared_error', 'friedman_mse',],
                                    'max_depth': [5, 10, 15, 20, 25, 30]
                                    ,'max_features': [None, 'sqrt', 'log2'],
                                    'max_depth': range(2, 5, 1),
                                    'min_samples_split': range(2, 4, 1),
                                 'splitter': ['best', 'random']
                                  }
        # Creating an object of the Grid Search class
        grid = GridSearchCV(self.dt_model, param_grid_decisionTree, verbose=3,cv=5)
        # finding the best parameters
        grid.fit(self.X_train, self.y_train)
    
        # extracting the best parameters
        criterion = grid.best_params_['criterion']
        splitter = grid.best_params_['splitter']
        max_features = grid.best_params_['max_features']
        max_depth  = grid.best_params_['max_depth']
        min_samples_split = grid.best_params_['min_samples_split']
    
        # creating a new model with the best parameters
        decisionTreeReg1 = DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_features=max_features,max_depth=max_depth,min_samples_split=min_samples_split)
        # training the mew models
        decisionTreeReg1.fit(self.X_train, self.y_train)
        predict_dt=decisionTreeReg1.predict(self.X_test)
        global dt_error
        self.dt_error=r2_score(self.y_test,predict_dt)
        return decisionTreeReg1
    
    def modal_build(self):
        #model={}
        st.write("Modal traing Started")
        list_of_clusters=self.X['Cluster'].unique() 
        sum=0
        
        for i in list_of_clusters:
            sum=sum+1
            st.write("Modal training for cluster " + str(sum) +" of " +str(self.s) + " started")
        
            cluster_data=self.X[self.X['Cluster']==i]
            cluster_data=cluster_data.drop("Cluster",axis=1)
            cluster_data_2=self.z[self.z['Cluster']==i]
            cluster_data2=cluster_data_2.drop("Cluster",axis=1)# filter the data for one cluster
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(cluster_data, cluster_data_2, test_size=0.2, random_state=42)
    
            scaler = StandardScaler()
            X_train = scaler.fit_transform(self.X_train)
            X_test = scaler.transform(self.X_test)
            
            folder_path = "standard scaler for cluster "+str(i)+".pkl"  # Use forward slash (/) for file paths in Python
            st.write(folder_path+" created")
            
            if os.path.exists(folder_path):
                os.remove(folder_path)
            #os.mkdir(folder_path)
            
            #file_path = os.path.join(folder_path, "cluster_modal.pkl")
            file = open(folder_path, 'wb')
            pickle.dump(scaler, file)
            file.close()
            
            st.write("training for cluster "+str(i))
            
            st.write("decision tree modal training begins")      
            dtdt=self.decession_tree_()
            st.write("decision tree modal training ends")
            st.write("XGBoost modal training begins") 
            xgbxgb=self.xg_boost_()
            st.write("XGBoost modal training ends")
            st.write("Random forest modal training begins") 
    
            rfrf=self.randomforest_()
            st.write("Random forest modal training ends") 
        
            self.error.append(self.randomForest_error)
            self.error.append(self.xgb_error)
            self.error.append(self.dt_error)
            #st.write(self.error)
    
            
            folder_path2 = "modal for cluster "+str(i)+".pkl"
            if os.path.exists(folder_path2):
                os.remove(folder_path2)
            #os.mkdir(folder_path2)
            
            
    
            if self.randomForest_error < self.xgb_error and self.randomForest_error < self.dt_error:
                st.write("Random forest selected for the cluster "+str(i))
                self.model[i]=rfrf
                #file_path = os.path.join(folder_path2, "modal.pkl")
                with open(folder_path2, 'wb') as file:
                    pickle.dump(rfrf, file)
    
            elif self.xgb_error < self.dt_error and self.xgb_error < self.randomForest_error:
                self.model[i]=xgbxgb
                #st.write(self.model[i])
                st.write("XGBoost selected for cluster "+str(i))
                #file_path = os.path.join(folder_path2, "modal.pkl")
                with open(folder_path2, 'wb') as file:
                    pickle.dump(xgbxgb, file)
              #  st.write("ok")
            else:
             #   st.write("78")
                self.model[i]=dtdt
                st.write("Decision tree selected for the cluster "+str(i))
                
                #file_path = os.path.join(folder_path2, "modal.pkl")
                with open(folder_path2, 'wb') as file:
                    pickle.dump(dtdt, file)
                
            
        st.write("Modal training completed")
        #st.write(self.model)




