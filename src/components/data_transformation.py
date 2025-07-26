import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

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
        self.all_categories = None
        self.encoder = None

    def fit(self, X, y=None):
        # Determine all possible categories (top categories + 'Other')
        self.all_categories = self.top_categories + [self.new_value]
        
        # Initialize and fit the OneHotEncoder with all possible categories
        self.encoder = OneHotEncoder(
            categories=[self.all_categories], 
            drop='first', 
            handle_unknown='ignore',
            sparse_output=False
        )
        
        # Create a temporary dataframe with all categories for fitting
        X_temp = X.copy()
        X_temp[self.column] = X_temp[self.column].apply(lambda x: x if x in self.top_categories else self.new_value)
        
        # Fit the encoder
        self.encoder.fit(X_temp[[self.column]])
        
        return self

    def transform(self, X, y=None):
        """Transforms the input data by replacing the less frequent categories with the new value and one-hot encoding the column."""
        X = X.copy()
        X[self.column] = X[self.column].apply(lambda x: x if x in self.top_categories else self.new_value)
        
        # Use sklearn OneHotEncoder for consistent encoding
        encoded_array = self.encoder.transform(X[[self.column]])
        
        # Get feature names from the encoder
        feature_names = self.encoder.get_feature_names_out([self.column])
        
        # Create DataFrame from encoded array
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=X.index)
        
        # Drop the original categorical column and add the encoded columns
        other_cols = [col for col in X.columns if col != self.column]
        result = pd.concat([X[other_cols], encoded_df], axis=1)
        
        return result
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if self.encoder is None:
            raise NotFittedError("This FixedTopCategoriesTransformer instance is not fitted yet.")
        
        if input_features is not None:
            other_features = [f for f in input_features if f != self.column]
        else:
            other_features = []
        
        # Get encoded feature names
        encoded_features = self.encoder.get_feature_names_out([self.column])
        
        return other_features + list(encoded_features)
    
class FloorNumberCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting process required for this cleaner
        return self

    def transform(self, X, y=None):
        # Handle both DataFrame and Series input
        if isinstance(X, pd.DataFrame):
            if 'floor_number' in X.columns:
                # DataFrame with floor_number column
                X_transformed = X['floor_number'].apply(self.clean_floor_number)
            else:
                # DataFrame but no floor_number column, assume the whole DataFrame is floor numbers
                X_transformed = X.iloc[:, 0].apply(self.clean_floor_number)
        else:
            # Series input
            X_transformed = X.apply(self.clean_floor_number)
        
        # Convert the Series to a DataFrame to ensure 2D output
        # Reset the name to avoid column name conflicts
        X_transformed.name = None
        return pd.DataFrame({'cleaned_floor_number': X_transformed}, index=X_transformed.index)

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
            # Convert to int to ensure consistent data type
            try:
                return int(floor_str)
            except (ValueError, TypeError):
                return np.nan

class DataTransformation:
    
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def pre_transform_floor_number(self, df):
        """Applies the FloorNumberCleaner transformation to the 'floor_number' column of the DataFrame."""
        cleaner = FloorNumberCleaner()
        # Transform returns a DataFrame with 'cleaned_floor_number' column
        transformed_floor_number = cleaner.transform(df['floor_number'])
        # Extract the cleaned values and assign back to floor_number column
        df['floor_number'] = transformed_floor_number['cleaned_floor_number']
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
                'balcony',
                ]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore')),
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
                'Södermalm, Stockholms kommun',
                'Vasastan, Stockholms kommun', 
                'Kungsholmen, Stockholms kommun',
                'Östermalm, Stockholms kommun',
                'Bromma, Stockholms kommun',
                'Årsta, Stockholms kommun',
                'Hammarby Sjöstad, Stockholms kommun',
                'Råsunda, Solna kommun',
                'Centrala Sundbyberg, Sundbybergs kommun',
                'Gröndal, Stockholms kommun',
                'Gärdet, Stockholms kommun',                
                'Huvudsta, Solna kommun',             
                'Kallhäll, Järfälla kommun',              
                'Jakobsberg, Järfälla kommun',          
                'Farsta, Stockholms kommun',            
                'Täby Centrum, Täby kommun',        
                'Liljeholmskajen, Stockholms kommun',   
                'Hammarbyhöjden, Stockholms kommun',    
                'Aspudden, Stockholms kommun',        
                'Barkarbystaden, Järfälla kommun',
                'Södermalm',
                'Vasastan',
                'Kungsholmen',
                'Östermalm',     
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("region_transformer", FixedTopCategoriesTransformer(column='region', top_categories=top_categories), ['region']),
                    ("floor_number_pipeline", floor_number_pipeline, ['floor_number']),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns),
                ],
                remainder='drop'
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
            