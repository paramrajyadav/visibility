### Atmospheric Visibility Prediction Project
This repository contains code and resources for a project focused on predicting atmospheric visibility based on historical data. 
The goal of this project is to develop a machine learning model that can effectively estimate visibility based on various environmental factors.

### Project Website: Interactive User Interface

We've developed an interactive user interface that allows you to easily interact with the project. 
The website provides options for both training on data and making predictions based on the trained model.

### Accessing the Website
To access the user interface, open the URL   

## https://visibility.streamlit.app/
The website offers the following functionalities:
### Train Model: 
This section allows you to upload your dataset and perform model training. It guides you through the steps required for data preprocessing, model selection, and training. Upon completion, the trained model is saved for later use.
### Make Predictions: 
In this section, you can provide input data and obtain predictions using the trained model. Simply follow the prompts to input your data and receive predictions.


### 1. Introduction:
In this project repository, I have developed a comprehensive workflow to perform data analysis, feature selection, model training, and predictions. The repository contains essential files that enable a seamless transition from data exploration to deploying a machine learning model for making predictions.

## 2. EDA File:
The EDA (Exploratory Data Analysis) file, named "EDA.ipynb," is a Jupyter Notebook that serves as the initial phase of the project. This file includes the following key tasks:

Data loading: Importing the dataset and loading it into memory.
Basic statistics: Calculating summary statistics (mean, median, standard deviation, etc.).
Data visualization: Creating visualizations like histograms, heat map, and box plots to understand data distributions and relationships.
Handling missing values: Identifying and addressing missing data points.
Outlier detection: Detecting outliers that might affect model performance.
Correlation analysis: Analyzing feature correlations to identify potential multicollinearity.

### 3. Feature Selection and Model Selection Process:
The process of selecting features and choosing an appropriate model, as detailed in the EDA file, encompassed the subsequent actions:

Statistical Assessment: Executing statistical examinations to pinpoint features that hold significant statistical relevance.

Leveraging Domain Insight: Applying expertise in the subject matter to cherry-pick pertinent features that contribute to the model's efficacy.

Feature Significance: Employing methodologies like feature importance scores derived from tree-based models.

Given the non-normally distributed nature of the data, I opted for tree-based models such as XGBoost, Random Forest, and Decision Tree for further consideration.

Model selection: Evaluating various machine learning algorithms and selecting the most appropriate one based on R2 score.

### 4. APP File:
The "app.py" file serves as the entry point for the application. This file is responsible for:

Setting up a user interface (UI) for users to interact with the model and input data.
Handling user inputs and passing them to the prediction module.
Displaying the prediction results back to the user through the UI.

### 5. Training.py:
The "training.py" script is dedicated to the training of the chosen machine learning model. The steps involved include:

Data loading and preprocessing: Loading the data and performing necessary preprocessing (outlier removal, standard scaler).
Data splitting: Dividing the dataset into training and validation sets for model training and evaluation.
Model initialization: Creating an instance of the chosen model architecture.
Model training: Fitting the model to the training data using appropriate training algorithms.
Model evaluation: Assessing the model's performance using validation data and relevant evaluation metrics (accuracy, F1-score, etc.).
Model saving: Saving the trained model's parameters or weights for future use.

### 6. Prediction.py:
The "prediction.py" script focuses on making predictions using the trained model. The steps include:

Loading the trained model: Loading the saved model from the training phase.
Data preprocessing: Preprocessing new input data to match the format expected by the model.
Model inference: Using the loaded model to predict outcomes for the input data.
Result presentation: Displaying the prediction results to the user or saving them to a file.


### 7. Conclusion:
This repository encapsulates the entire data science workflow, from initial data exploration to deploying a functional machine learning model. The modular structure of files ensures a clear separation of tasks, making it easy to understand, modify, and expand upon the project.

### 8. Usage Instructions:
To run the project, follow the instructions provided in the README file. Ensure you have the necessary dependencies installed and execute the appropriate scripts in the correct order. The README contains step-by-step guidelines to reproduce the results and predictions.

### Getting Started
To run this project locally, you'll need to set up a Python environment and install the required dependencies listed in the requirements.txt file.


