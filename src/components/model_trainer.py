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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer initiated.")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            model = {
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier()
            }
            
            params = {
                "Random Forest": {
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced']
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy','log_loss'],
                    'max_depth': [3,4,5,6],
                    'min_samples_split': [2,3,4,5],
                    'class_weight': ['balanced'],
                    'splitter': ['best', 'random'],
                    'min_samples_leaf': [1,2,3,]

                },
                "Logistic Regression": {
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': [ 'liblinear', 'saga'],
                    'class_weight': ['balanced']
                }

            }
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, model, params)
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] 

            best_model = model[best_model_name]
            best_model.fit(X_train, y_train)
            logging.info(f"Best model found on both training and testing dataset is {best_model_name} with accuracy score: {best_model_score}") 

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            return best_model_name
        
        except Exception as e:
            logging.info("Error occurred in model trainer component.")
            raise CustomException(e, sys)