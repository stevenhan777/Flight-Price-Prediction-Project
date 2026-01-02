import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.exception import CustomException # import from exception.py
from src.logger import logging 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass # decorator allow class to directly define vars
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") # path where saved
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/cleaned_for_user_input.csv') #, keep_default_na=False) # to prevent NaN in 'Route' columns
            logging.info('Read the dataset as dataframe')

            df.dropna(inplace=True)
            df = df.drop_duplicates() 
            logging.info('Remove NA and Duplicates from the dataset')

            df['Source'] = "S_" + df['Source']
            df['Destination'] = "D_" + df['Destination']
            logging.info('Prefaced the Source and Destination columns with S_ and D_ respectively')

            df = df[df['Price'] <= 40000]
            logging.info('Removed prices greater than 40000 as outliers')

            df['Total_Stops']=df['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3})
            logging.info('Mapped Total_Stops column to numerical values')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train and test data split started')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()

    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    #train_arr.to_csv('train_array.csv', index=False, header=True)
    #np.savetxt('data.csv', train_arr, delimiter=',')
    np.savetxt('data.csv', train_arr, delimiter=',')

    modeltrainer=ModelTrainer()
    best_model_name, r2_square = modeltrainer.initiate_model_trainer(train_arr,test_arr)
    print(best_model_name)
    print(r2_square)
































# import os
# import sys

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# from src.exception import CustomException # import from exception.py
# from src.logger import logging 
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

# @dataclass # decorator allow class to directly define vars
# class DataIngestionConfig:
#     train_data_path: str=os.path.join('artifacts',"train.csv") # path where saved
#     test_data_path: str=os.path.join('artifacts',"test.csv")
#     raw_data_path: str=os.path.join('artifacts',"data.csv")

# class DataIngestion:
#     def __init__(self):
#         self.ingestion_config=DataIngestionConfig()

#     def initiate_data_ingestion(self):
#         logging.info("Entered the data ingestion method or component")
#         try:
#             df=pd.read_excel('notebook/data/flight_price_pred.xlsx')
#             logging.info('Read the dataset as dataframe')

#             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

#             df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

#             logging.info("Ingestion of the data is completed")

#             return(
#                 self.ingestion_config.raw_data_path
#             )
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def save_train_test(self,train_set,test_set):
#         logging.info("Entered the save_train_test")
#         try:
#             train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

#             test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

#             logging.info("test and train data saved to artifacts")

#             return(
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path

#             )
#         except Exception as e:
#             raise CustomException(e,sys)
        
# if __name__=="__main__":
#     obj=DataIngestion()
#     raw_data = obj.initiate_data_ingestion()

#     data_transformation=DataTransformation()
#     train_arr,test_arr,train_df,test_df,_=data_transformation.initiate_data_transformation() # train_data,test_data

#     DataIngestion().save_train_test(train_df,test_df)

#     modeltrainer=ModelTrainer()
#     best_model_name, r2score = modeltrainer.initiate_model_trainer(train_arr,test_arr)
#     print(f"best model: {best_model_name}")
#     print(f"R2 Score: {r2score}")


