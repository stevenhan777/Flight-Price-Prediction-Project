import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

import numpy as np


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Get the absolute path to the project root directory
            # Go up from src/pipeline/ to the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))

            model_path = os.path.join(project_root, "artifacts", "model.pkl")
            preprocessor_path = os.path.join(project_root, 'artifacts', 'preprocessor.pkl')
            
            print(f"Current directory: {current_dir}")
            print(f"Project root: {project_root}")
            print(f"Loading model from: {model_path}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            print(f"Loading preprocessor from: {preprocessor_path}")
            print(f"Preprocessor file exists: {os.path.exists(preprocessor_path)}")
            
            # List contents of artifacts directory
            artifacts_dir = os.path.join(project_root, "artifacts")
            if os.path.exists(artifacts_dir):
                print(f"Contents of artifacts directory: {os.listdir(artifacts_dir)}")
            else:
                print(f"Artifacts directory does not exist at: {artifacts_dir}")

            print("Before Loading")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Preprocessor loaded successfully")
            
            # print("After Loading")
            # data_scaled = preprocessor.transform(features)
            # #data_scaled = features
            # preds = model.predict(data_scaled)



             
            print("After Loading")
            print(f"Input features shape before preprocessing: {features.shape}")
            print(f"Input features:\n{features}")
            
            # Apply the same preprocessing transformation used during training
            data_scaled = preprocessor.transform(features)
            print(f"Data shape after preprocessing: {data_scaled.shape}")
            
            # Get predictions (these will be in log scale since target was log-transformed)
            preds_log = model.predict(data_scaled)
            print(f"Log-transformed predictions: {preds_log}")
            
            # Inverse transform from log scale back to actual price
            # Since we used np.log1p() during training, we use np.expm1() to reverse it
            preds = np.expm1(preds_log)
            print(f"Final predictions (actual price): {preds}")


            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        Airline: str,
        Source: str,
        Destination: str,
        Total_Stops: int,
        dept_day: int,
        dept_month: int,
        dept_hour: int,
        dept_minute: int,
        arrival_day: int,
        arrival_month: int,
        arrival_hour: int,
        arrival_minute: int,
        duration_hours: int,
        duration_minutes: int,
        Route1: str,
        Route2: str,
        Route3: str,
        Route4: str,
        Route5: str):

        self.Airline = Airline
        self.Source = Source
        self.Destination = Destination
        self.Total_Stops = Total_Stops
        self.dept_day = dept_day
        self.dept_month = dept_month
        self.dept_hour = dept_hour
        self.dept_minute = dept_minute
        self.arrival_day = arrival_day
        self.arrival_month = arrival_month
        self.arrival_hour = arrival_hour
        self.arrival_minute = arrival_minute
        self.duration_hours = duration_hours
        self.duration_minutes = duration_minutes
        self.Route1 = Route1
        self.Route2 = Route2
        self.Route3 = Route3
        self.Route4 = Route4
        self.Route5 = Route5

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Airline": [self.Airline],
                "Source": [self.Source],
                "Destination": [self.Destination],
                "Total_Stops": [self.Total_Stops],
                "dept_day": [self.dept_day],
                "dept_month": [self.dept_month],
                "dept_hour": [self.dept_hour],
                "dept_minute": [self.dept_minute],
                "arrival_day": [self.arrival_day],
                "arrival_month": [self.arrival_month],
                "arrival_hour": [self.arrival_hour],
                "arrival_minute": [self.arrival_minute],
                "duration_hours": [self.duration_hours],
                "duration_minutes": [self.duration_minutes],
                "Route1": [self.Route1],
                "Route2": [self.Route2],
                "Route3": [self.Route3],
                "Route4": [self.Route4],
                "Route5": [self.Route5]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)