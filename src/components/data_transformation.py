import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated.")
            numerical_features = ['ID','Age','Education','Gender']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed.")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.info("Error occurred in data transformation component.")
            raise CustomException(e, sys)
        
    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound 
            return df
        
        except Exception as e:
            logging.info("Outliers handling code.")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            numerical_features = ['ID','Age','Education','Gender']

            for i in numerical_features:
             self.remove_outliers_IQR(col = i, df = train_df)
            
            logging.info("Outliers capping completed for training dataset.")  
            
            for i in numerical_features:  
             self.remove_outliers_IQR(col = i, df = test_df)
            
            logging.info("Outliers capping completed for testing dataset.")

            preprocess_obj = self.get_data_transformer_object()
            target_column_name = "Income"
            drop_columns = [target_column_name, "Income"]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            #Apply transformation on training and testing datasets
            input_train_arr = preprocess_obj.fit_transform(input_feature_train_df)
            input_test_arr = preprocess_obj.transform(input_feature_test_df)

            #apply preprocessor obj on training and testing datasets
            train_arr = np.c_[input_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocess_obj)

            return(train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            logging.info("Error occurred")
            raise CustomException(e, sys)