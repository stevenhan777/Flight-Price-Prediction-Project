# Flight Price Prediction Project

### Life cycle of Machine learning Project:

* Understanding the Problem Statement
* Data Collection
* Data Checks to perform
* Exploratory data analysis
* Data Pre-Processing, feature engineering and scaling
* Model Training
* Feature selection
* Hyperparameter Tuning
* Best model
* Flask web app
* AWS Deployment

#### 1) Problem Statement

This project aims to understand and model how different features of a flight, such as arrival/departure city, number of stops, flight duration, flight departure time, etc, affect the flight price.

#### 2) Data Collection

The data is obtained from Kaggle: https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh?resource=download

#### 3) EDA Notebook

Data cleaning, exploratory data analysis and feature engineering is performed. Many of the columns were in inconsistent format and required extensive feature engineering. Highly correlated features and features with low mutual information gain are removed. The dataset is split into train and test split. The dataset is scaled with MinMax on the features and with log scale on the dependent variable. The dataset is also left unscaled for testing. 

#### 4) Model Training Notebook

The models are fitted on the scaled and unscaled datasets and determined that the scaled dataset has the best overall results and R2 score. The best performing models include Random Forest, CatBoost, XGBoost, and Gradient Boost, with CatBoost having the best R2 score. 

#### 5) Feature Selection Notebook

Perform feature selection using the CatBoost model on the scaled dataset. Try iterative feature removal that removes 2 features at a time. Then try Recursive feature elimination with cross-validation (RFECV) to select optimal features based off adjusted R2 score. End result is removing 1 feature with the lowest feature importance score. 

#### 6) Hyperparameter Tuning Notebook

Here, I perform quick hyperparameter tuning on the most promising models: Decision Tree, Random Forest, K-Neighbors, XGBoost, CatBoost, and GradientBoost. Then I perform more comprehensive hyperparamter tuning on the most promising models: XGBoost, CatBoost, and GradientBoost. Final result is the CatBoost model after hyperparameter tuning has the best R2 score.  

#### 7) Modular implementation

I then implemented my process in a modular way within the components folder with a data_ingestion, data_transformation and model_trainer steps. In the Notebook 0: Input Dataset I perform preliminary data cleaning and feature engineering to transform dataset into a user friendly input. Then I further transform the input data with Column Transformer and a pipeline and perform hyperparameter tuning to determine the best model. 

#### 8) Flask web app implementation

I then create a predict_pipeline, application and home.html file to create a simple web application for the user to input the fields and get a prediction of the flight price using the best model pickle file. I added user input constraints to make sure the user entered fields are accurate. 

#### 9) Deployment on AWS

I then use AWS Elastic Beanstalk and CodePipeline to deploy the Flask web app. The user can successfully predict flight price using the best model. Deployment Link: http://flightpriceprediction-env.eba-w99s2cmh.us-east-1.elasticbeanstalk.com/predictdata

#### Conclusion:

This project successfully developed and deployed a machine learning regression model for flight price prediction through comprehensive feature engineering and data exploration. I analyzed which features had a greater effect on the target variable and implemented appropriate scaling techniques.With feature selection and hyperparameter tuning, I determined the best model and deployed using AWS. Successful deployment involved consistent versioning and ensuring an intuitive user experience. This project provides price predictions that can help travelers make informed booking decisions.