import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        model_path = os.path.join('artifacts', 'model_2024_03_11.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

        self.model = load_object(filepath = model_path)
        self.preprocessor = load_object(filepath = preprocessor_path)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            prediction = self.model.predict(data_scaled)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """Map form data to the ML Model"""

    def __init__(
            self,
            number_of_rooms: int,
            area_size: int,
            year_built: int,
            annual_fee_sek: int,
            region: str,
            has_balcony: str,
            floor_number: int,
    ):
        self.number_of_rooms = number_of_rooms
        self.area_size = area_size
        self.year_built = year_built
        self.annual_fee_sek = annual_fee_sek
        self.region = region
        self.has_balcony = has_balcony
        self.floor_number = floor_number

    def get_data_as_datafrane(self):
        try:
            return pd.DataFrame({
                'number_of_rooms': [self.number_of_rooms],
                'area_size': [self.area_size],
                'year_built': [self.year_built],
                'annual_fee_sek': [self.annual_fee_sek],
                'region': [self.region],
                'has_balcony': [1 if self.has_balcony == 'yes' else 0],
                'floor_number': [self.floor_number],
            })
        except Exception as e:
            raise CustomException(e, sys)