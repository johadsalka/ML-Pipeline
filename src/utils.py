
import os, sys
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import accuracy_score,confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.info("Error occurred in saving object.")
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            gs = GridSearchCV(model, param, cv=5)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_score = accuracy_score(y_test, y_pred)
            report[list(models.keys())[i]] = acc_score
        return report
    except Exception as e:
        logging.info("Error occurred in evaluating model.")
        raise CustomException(e, sys)