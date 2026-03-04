import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
  train_file_path = os.path.join("artifacts", "train.csv")
  test_file_path = os.path.join("artifacts", "test.csv")
  raw_data_path = os.path.join("artifacts", "raw.csv")

#notebook\data\income.csv

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")
        try:
            df = pd.read_csv(os.path.join("notebook/data", "income.csv"))
            logging.info("Read the dataset as dataframe.")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated.")
            

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_file_path, index=False, header=True)

            logging.info("Ingestion of the data is completed.")

            return (
                self.ingestion_config.train_file_path,
                self.ingestion_config.test_file_path
            )
        except Exception as e:
            logging.info("Error occurred in data ingestion component.")
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

#src\components\data_ingestion.py