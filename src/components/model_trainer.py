import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], #take out last column
                train_array[:,-1], # use last column
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                # "Random Forest": RandomForestRegressor(),
                # "Decision Tree": DecisionTreeRegressor(),
                # "Gradient Boosting": GradientBoostingRegressor(),
                # "K-Neighbors Regressor": KNeighborsRegressor(),
                # "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(random_state = 42,verbose=False)
            }

            params={
                # "Random Forest":{
                #     'n_estimators': [100, 200],
                #     'max_depth': [10, 20, None],
                #     'min_samples_split': [2, 5],
                #     'min_samples_leaf': [1, 2],
                #     'max_features': ['sqrt', 'log2']
                # },
                # "Decision Tree": {
                #     'max_depth': [5, 10, 15, 20, None],
                #     'min_samples_split': [2, 5, 10],
                #     'min_samples_leaf': [1, 2, 4],
                #     'max_features': ['sqrt', 'log2', None]
                # },
                # "Gradient Boosting":{
                #     'n_estimators': [100, 200, 300, 500],
                #     'max_depth': [3, 5, 7, 9, 11],
                #     'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
                #     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                #     'min_samples_split': [2, 5, 10, 15],
                #     'min_samples_leaf': [1, 2, 4, 6],
                #     'max_features': ['sqrt', 'log2', None],
                #     'loss': ['squared_error', 'absolute_error', 'huber']
                # },
                # "K-Neighbors Regressor":{
                #     'n_neighbors': [3, 5, 7, 9],
                #     'weights': ['uniform', 'distance'],
                #     'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                #     'leaf_size': [20, 30, 40]
                # },
                # "XGBRegressor":{
                #     'n_estimators': [100, 200, 300, 500],
                #     'max_depth': [3, 5, 7, 9, 11],
                #     'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
                #     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                #     'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                #     'gamma': [0, 0.1, 0.3, 0.5, 1],
                #     'min_child_weight': [1, 3, 5, 7],
                #     'reg_alpha': [0, 0.01, 0.1, 1],
                #     'reg_lambda': [0.1, 1, 10]
                # },
                "CatBoosting Regressor":{
                    'iterations': [100, 200, 300, 500],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128],
                    'subsample': [0.7, 0.8, 0.9, 1.0]
                }
            }

            # Model report function
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params) 
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values())) # sort based on valuse

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return (best_model_name,
                    r2_square
            )
                        
        except Exception as e:
            raise CustomException(e,sys)
            