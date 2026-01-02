import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
import category_encoders as ce

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:

            target_encode_columns = [
                "Route1", "Route2", "Route3", "Route4", "Route5"
            ]
            one_hot_encode_columns = [
                "Airline",
                "Source",
                "Destination"
            ]
            columns_to_drop = [
                "D_Cochin",
                "D_Hyderabad",
                "D_Kolkata",
                "arrival_day",
                "arrival_month",
                "Route5",
                "Jet Airways Business",
                "Multiple carriers Premium economy",
                "Trujet",
                "Vistara Premium economy"
            ]

            def drop_columns(X, columns_to_drop):
                if isinstance(X, pd.DataFrame):
                    existing_cols = [col for col in columns_to_drop if col in X.columns]
                    return X.drop(columns=existing_cols, errors='ignore')
                return X

            feature_transformer = ColumnTransformer(
                transformers=[
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                    one_hot_encode_columns),
                    ('target_encoder', ce.TargetEncoder(), target_encode_columns)],
                remainder='passthrough'
            )
            
            preprocessor = Pipeline(steps=[
                ('feature_transformer', feature_transformer),
                ('drop_columns', FunctionTransformer(
                    drop_columns, 
                    kw_args={'columns_to_drop': columns_to_drop}
                )),
                ('minmax_scaling', MinMaxScaler())
            ])

            logging.info("Pipeline created to One hot encode, target encode, drop unneeded columns and MinMax scale")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name="Price"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training data only")

            preprocessing_obj.fit(input_feature_train_df, target_feature_train_df)

            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)

            #input_feature_train_arr = drop_cols_preprocessor.fit_transform(input_feature_train_arr)
            target_feature_train_arr = np.log1p(target_feature_train_df)
            logging.info("Transforming training features and target completed")

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            #input_feature_test_arr = drop_cols_preprocessor.fit_transform(input_feature_test_arr)
            
            target_feature_test_arr = np.log1p(target_feature_test_df)
            logging.info("Transforming test features and target completed")



            train_arr = np.c_[
                input_feature_train_arr, target_feature_train_arr
            ]
            test_arr = np.c_[
                input_feature_test_arr, target_feature_test_arr
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)