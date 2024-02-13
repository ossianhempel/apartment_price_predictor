import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], # select all rows and all columns except the last one (target values)
                train_array[:, -1], # select all rows and the last column (target values)
                test_array[:, :-1], 
                test_array[:, -1]
                )
            
            # TODO: add parameter tuning to the training pipeline

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            model_report: dict=evaluate_model(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models
                )
            
            best_model_name = max(model_report, key=model_report.get)  # This directly gives you the model name with the highest score
            best_model_score = model_report[best_model_name]  # Directly access the score using the model name
            best_model = models[best_model_name]  # Access the model instance using the model name

            if best_model_score < 0.6:
                raise CustomException("Best model score is less than 0.6")
            
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return best_model, r2_square

        except Exception as e:
            raise CustomException(e, sys)