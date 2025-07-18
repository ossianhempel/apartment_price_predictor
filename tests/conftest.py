"""
Pytest configuration and fixtures for apartment price predictor tests.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch

# Add src to Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_apartment_data():
    """Create sample apartment data for testing."""
    data = {
        'number_of_rooms': [2, 3, 4, 2, 3],
        'area_size': [65, 85, 110, 55, 90],
        'year_built': [2000, 1995, 2010, 1985, 2005],
        'annual_fee_sek': [35000, 45000, 55000, 25000, 40000],
        'region': ['Södermalm, Stockholms kommun', 'Östermalm, Stockholms kommun', 
                  'Vasastan, Stockholms kommun', 'Södermalm, Stockholms kommun',
                  'Kungsholmen, Stockholms kommun'],
        'balcony': ['Ja', 'Nej', 'Ja', 'Unknown', 'Ja'],
        'floor_number': [2, 5, 3, 1, 4],
        'price_sold_sek': [4500000, 6200000, 8500000, 3800000, 5900000]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_prediction_data():
    """Create sample data for prediction testing."""
    return {
        'number_of_rooms': 3,
        'area_size': 80,
        'year_built': 2010,
        'annual_fee_sek': 50000,
        'region': 'södermalm',
        'has_balcony': 'yes',
        'floor_number': 5
    }


@pytest.fixture
def temp_artifacts_dir():
    """Create a temporary artifacts directory for testing."""
    temp_dir = tempfile.mkdtemp()
    artifacts_dir = os.path.join(temp_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    yield artifacts_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_trained_model():
    """Create a mock trained model for testing."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    # Create some dummy data to fit the model
    X = np.random.rand(100, 31)  # 31 features to match our pipeline
    y = np.random.rand(100) * 1000000  # Random prices
    model.fit(X, y)
    return model


@pytest.fixture
def mock_preprocessor():
    """Create a mock preprocessor for testing."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Create a simple mock preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, list(range(31)))  # All 31 features as numeric for mock
        ]
    )
    
    # Fit with dummy data
    X_dummy = np.random.rand(100, 31)
    preprocessor.fit(X_dummy)
    
    return preprocessor


@pytest.fixture
def sample_csv_files(temp_artifacts_dir, sample_apartment_data):
    """Create sample CSV files for testing."""
    train_path = os.path.join(temp_artifacts_dir, 'train.csv')
    test_path = os.path.join(temp_artifacts_dir, 'test.csv')
    
    # Split data into train/test
    train_data = sample_apartment_data.iloc[:4]
    test_data = sample_apartment_data.iloc[4:]
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    return train_path, test_path