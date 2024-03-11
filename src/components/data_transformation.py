import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

# TODO - dealing with outliers? or in data validation?
# TODO - dealing with missing values?
# TODO - dealing with multicollinearity? or in data validation?
    

class FixedTopCategoriesTransformer(BaseEstimator, TransformerMixin):
    """Transforms a categorical column by keeping only the predefined top categories and replacing the rest with 'Other', then applying one-hot encoding."""

    def __init__(self, column, top_categories, new_value='Other'):
        self.column = column
        # Predefined list of top categories
        self.top_categories = top_categories
        self.new_value = new_value

    def fit(self, X, y=None):
        # No fitting process required as top categories are predefined
        return self

    def transform(self, X, y=None):
        """Transforms the input data by replacing the less frequent categories with the new value and one-hot encoding the column."""
        X = X.copy()
        X[self.column] = X[self.column].apply(lambda x: x if x in self.top_categories else self.new_value)
        return pd.get_dummies(X, columns=[self.column], drop_first=True)
    
class FloorNumberCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting process required for this cleaner
        return self

    def transform(self, X, y=None):
        # Assuming X is a DataFrame with a 'floor_number' column
        if isinstance(X, pd.DataFrame):
            X_transformed = X.apply(lambda row: self.clean_floor_number(row['floor_number']), axis=1)
        else:
            X_transformed = X.apply(self.clean_floor_number)
        
        # Convert the Series to a DataFrame to ensure 2D output
        return pd.DataFrame(X_transformed, columns=['cleaned_floor_number'])

    def clean_floor_number(self, floor_str):
        # First, check if floor_str is a string
        if isinstance(floor_str, str):
            # Process the string to extract floor number
            for part in floor_str.split():
                if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                    return int(part)
            return np.nan  # Return NaN if no numeric part is found
        elif pd.isna(floor_str):
            # If floor_str is NaN (missing value), return it as is
            return np.nan
        else:
            # For numeric types, return the value directly. This handles already cleaned or numeric inputs.
            return floor_str

class DataTransformation:
    
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def pre_transform_floor_number(self, df):
        """Applies the FloorNumberCleaner transformation to the 'floor_number' column of the DataFrame."""
        cleaner = FloorNumberCleaner()
        # Assuming 'transform' method is adjusted to return a Series or compatible format
        transformed_floor_number = cleaner.transform(df['floor_number'])
        df['floor_number'] = transformed_floor_number  # Directly modify 'floor_number'
        return df

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation. 
        It will return the preprocessor object.
        """

        """
        The categorical variables are too sparse so there are categories
        in the training data that doesn't exist in the test data and vice versa.
        This will cause the one hot encoder to create different number of columns
        """

        try:
            numerical_columns = [
                'number_of_rooms', 
                'area_size', 
                'year_built',
                'annual_fee_sek', 
                ]
            # add 'cleaned_floor_number' dynamically based on your transformation logic
            #numerical_columns.append('cleaned_floor_number')
            
            categorical_columns = [
                'region', 
                'has_balcony',
                ]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            # TODO - am i going to use this? what about has_balcony?
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # this includes both your custom cleaning and an imputation step
            floor_number_pipeline = Pipeline(steps=[
                ('cleaner', FloorNumberCleaner()),
                ('imputer', SimpleImputer(strategy='median'))  # Or choose an appropriate imputation strategy
            ])


            top_categories = [
                'södermalm',
                'vasastan',
                'kungsholmen',
                'östermalm',
                'bromma',
                'årsta',
                'hammarby sjöstad',
                'råsunda',
                'centrala sundbyberg',
                'gröndal',
                'gärdet',                
                'huvudsta',             
                'kallhäll',              
                'jakobsberg',          
                'farsta',            
                'täby centrum',        
                'liljeholmskajen',   
                'hammarbyhöjden',    
                'aspudden',        
                'barkarbystaden',     
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("region_transformer", FixedTopCategoriesTransformer(column='region', top_categories=top_categories), ['region']),
                    ("floor_number_pipeline", floor_number_pipeline, ['floor_number']),
                    #("categorical_pipeline", categorical_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for initiating the data transformation process.
        It reads the train and test data from the given file paths.
        It applies the preprocessor object to transform the input features.
        It saves the preprocessor object to a file.
        
        Args:
            train_path (str): The file path of the train data.
            test_path (str): The file path of the test data.
        
        Returns:
            tuple: A tuple containing the transformed train and test data arrays, and the file path of the saved preprocessor object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Pre-transform 'floor_number' before the main transformation
            train_df = self.pre_transform_floor_number(train_df)
            test_df = self.pre_transform_floor_number(test_df)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column = "price_sold_sek"
            
            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessor object to train and test data")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
            