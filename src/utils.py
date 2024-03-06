import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

import numpy as np
import pandas as pd

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        # Wrap the models iteration with tqdm for a progress bar
        for model_name, model in tqdm(models.items(), desc="Evaluating Models"):
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Once the best parameters are found, retrain the model on the full training set
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate the R^2 score as an example metric
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            # Optional: Log the progress with the score
            tqdm.write(f"Completed {model_name} with test score: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(filepath):
    try:
        with open(filepath, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)