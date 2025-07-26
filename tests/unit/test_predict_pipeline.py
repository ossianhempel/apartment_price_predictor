"""
Unit tests for the prediction pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock

from src.pipeline.predict_pipeline import PredictPipeline, CustomData


class TestCustomData:
    """Test suite for CustomData class."""
    
    def test_custom_data_initialization(self):
        """Test CustomData initialization with valid inputs."""
        data = CustomData(
            number_of_rooms=3,
            area_size=80,
            year_built=2010,
            annual_fee_sek=50000,
            region='södermalm',
            has_balcony='yes',
            floor_number=5
        )
        
        assert data.number_of_rooms == 3
        assert data.area_size == 80
        assert data.year_built == 2010
        assert data.annual_fee_sek == 50000
        assert data.region == 'södermalm'
        assert data.has_balcony == 'yes'
        assert data.floor_number == 5
    
    def test_get_data_as_dataframe_structure(self):
        """Test that get_data_as_dataframe returns correct structure."""
        data = CustomData(
            number_of_rooms=3,
            area_size=80,
            year_built=2010,
            annual_fee_sek=50000,
            region='södermalm',
            has_balcony='yes',
            floor_number=5
        )
        
        df = data.get_data_as_datafrane()  # Note: typo in original method name
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Single row
        assert len(df.columns) == 7  # Seven features
        
        expected_columns = [
            'number_of_rooms', 'area_size', 'year_built', 
            'annual_fee_sek', 'region', 'balcony', 'floor_number'
        ]
        assert list(df.columns) == expected_columns
    
    def test_balcony_mapping(self):
        """Test that balcony values are correctly mapped."""
        # Test 'yes' mapping
        data_yes = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
        df_yes = data_yes.get_data_as_datafrane()
        assert df_yes['balcony'].iloc[0] == 'Ja'
        
        # Test 'no' mapping
        data_no = CustomData(3, 80, 2010, 50000, 'södermalm', 'no', 5)
        df_no = data_no.get_data_as_datafrane()
        assert df_no['balcony'].iloc[0] == 'Nej'
        
        # Test unknown mapping
        data_unknown = CustomData(3, 80, 2010, 50000, 'södermalm', 'maybe', 5)
        df_unknown = data_unknown.get_data_as_datafrane()
        assert df_unknown['balcony'].iloc[0] == 'Unknown'
    
    def test_region_mapping(self):
        """Test that region values are correctly mapped."""
        test_cases = [
            ('södermalm', 'Södermalm, Stockholms kommun'),
            ('östermalm', 'Östermalm, Stockholms kommun'),
            ('vasastan', 'Vasastan, Stockholms kommun'),
            ('unknown_region', 'unknown_region')  # Should remain unchanged
        ]
        
        for input_region, expected_region in test_cases:
            data = CustomData(3, 80, 2010, 50000, input_region, 'yes', 5)
            df = data.get_data_as_datafrane()
            assert df['region'].iloc[0] == expected_region
    
    def test_data_types(self):
        """Test that data types are preserved correctly."""
        data = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
        df = data.get_data_as_datafrane()
        
        assert df['number_of_rooms'].dtype in ['int64', 'int32']
        assert df['area_size'].dtype in ['int64', 'int32']
        assert df['year_built'].dtype in ['int64', 'int32']
        assert df['annual_fee_sek'].dtype in ['int64', 'int32']
        assert df['region'].dtype == 'object'
        assert df['balcony'].dtype == 'object'
        assert df['floor_number'].dtype in ['int64', 'int32']


class TestPredictPipeline:
    """Test suite for PredictPipeline class."""
    
    @patch('src.pipeline.predict_pipeline.load_object')
    @patch('os.path.join')
    def test_predict_pipeline_initialization(self, mock_join, mock_load_object):
        """Test PredictPipeline initialization."""
        # Setup mocks
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Initialize pipeline
        pipeline = PredictPipeline()
        
        # Assertions
        assert pipeline.model == mock_model
        assert pipeline.preprocessor == mock_preprocessor
        assert mock_load_object.call_count == 2
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_predict_method_success(self, mock_load_object):
        """Test successful prediction."""
        from unittest.mock import Mock
        
        # Create mock objects
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_model.predict.return_value = np.array([5000000.0])
        mock_preprocessor.transform.return_value = np.random.rand(1, 31)
        
        # Setup mocks
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Create pipeline
        pipeline = PredictPipeline()
        
        # Create test data
        test_data = pd.DataFrame({
            'number_of_rooms': [3],
            'area_size': [80],
            'year_built': [2010],
            'annual_fee_sek': [50000],
            'region': ['Södermalm, Stockholms kommun'],
            'balcony': ['Ja'],
            'floor_number': [5]
        })
        
        # Execute prediction
        result = pipeline.predict(test_data)
        
        # Assertions
        assert isinstance(result, np.ndarray)
        assert len(result) == 1  # Single prediction
        mock_preprocessor.transform.assert_called_once_with(test_data)
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_predict_method_failure(self, mock_load_object):
        """Test prediction failure handling."""
        # Setup mocks to raise exception
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.transform.side_effect = Exception("Transformation failed")
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Create pipeline
        pipeline = PredictPipeline()
        
        # Create test data
        test_data = pd.DataFrame({'col1': [1]})
        
        # Execute and assert exception
        with pytest.raises(Exception):
            pipeline.predict(test_data)
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_predict_with_custom_data_integration(self, mock_load_object):
        """Test prediction with CustomData integration."""
        from unittest.mock import Mock
        
        # Create mock objects
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_model.predict.return_value = np.array([5000000.0])
        mock_preprocessor.transform.return_value = np.random.rand(1, 31)
        
        # Setup mocks
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Create pipeline
        pipeline = PredictPipeline()
        
        # Create CustomData
        custom_data = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
        df = custom_data.get_data_as_datafrane()
        
        # Execute prediction
        result = pipeline.predict(df)
        
        # Assertions
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        mock_preprocessor.transform.assert_called_once()


class TestPredictionWorkflow:
    """Test the complete prediction workflow."""
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_end_to_end_prediction_workflow(self, mock_load_object):
        """Test complete workflow from CustomData to prediction."""
        from unittest.mock import Mock
        
        # Create mock objects
        mock_model = Mock()
        mock_preprocessor = Mock()
        expected_prediction = np.array([5000000.0])  # 5M SEK
        mock_model.predict.return_value = expected_prediction
        mock_preprocessor.transform.return_value = np.random.rand(1, 31)
        
        # Setup mocks
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Create pipeline
        pipeline = PredictPipeline()
        
        # Test different scenarios
        test_scenarios = [
            {
                'name': 'Södermalm apartment with balcony',
                'data': CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
            },
            {
                'name': 'Östermalm apartment without balcony',
                'data': CustomData(4, 120, 2015, 60000, 'östermalm', 'no', 3)
            },
            {
                'name': 'Unknown region',
                'data': CustomData(2, 60, 2000, 40000, 'unknown_place', 'yes', 2)
            }
        ]
        
        for scenario in test_scenarios:
            df = scenario['data'].get_data_as_datafrane()
            result = pipeline.predict(df)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == 1
            # The mock will return the same value, but in real scenario they'd differ
            assert result[0] == expected_prediction[0]
    
    def test_data_validation_in_custom_data(self):
        """Test that CustomData handles edge cases properly."""
        # Test with extreme values
        extreme_data = CustomData(
            number_of_rooms=10,  # Very large apartment
            area_size=300,
            year_built=1900,  # Very old
            annual_fee_sek=100000,  # High fee
            region='södermalm',
            has_balcony='yes',
            floor_number=20  # High floor
        )
        
        df = extreme_data.get_data_as_datafrane()
        
        # Should not raise exception and return valid DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df['number_of_rooms'].iloc[0] == 10
        assert df['area_size'].iloc[0] == 300
    
    def test_region_case_sensitivity(self):
        """Test that region mapping handles case sensitivity."""
        test_cases = [
            'SÖDERMALM',
            'Södermalm', 
            'södermalm',
            'SöDErMaLm'
        ]
        
        for region_case in test_cases:
            data = CustomData(3, 80, 2010, 50000, region_case, 'yes', 5)
            df = data.get_data_as_datafrane()
            # Should map to the same standard format regardless of case
            expected = 'Södermalm, Stockholms kommun'
            assert df['region'].iloc[0] == expected


if __name__ == "__main__":
    pytest.main([__file__])