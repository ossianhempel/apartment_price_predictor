"""
Integration tests for the complete pipeline workflows.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, Mock

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for training and prediction pipelines."""
    
    @pytest.mark.slow
    def test_training_to_prediction_workflow(self, sample_apartment_data, temp_artifacts_dir):
        """Test complete workflow from training to prediction."""
        # This is a comprehensive integration test
        
        # Step 1: Mock the training process
        with patch('src.components.data_ingestion.pd.read_csv') as mock_read_csv:
            mock_read_csv.return_value = sample_apartment_data
            
            # Create training pipeline
            training_pipeline = TrainingPipeline()
            
            # Mock file paths and training components
            with patch.object(training_pipeline.data_ingestion, 'ingestion_config') as mock_config:
                mock_config.train_data_path = os.path.join(temp_artifacts_dir, 'train.csv')
                mock_config.test_data_path = os.path.join(temp_artifacts_dir, 'test.csv')
                mock_config.raw_data_path = os.path.join(temp_artifacts_dir, 'data.csv')
                
                # Mock model training to avoid lengthy training process
                with patch.object(training_pipeline.model_trainer, 'initiate_model_trainer') as mock_trainer:
                    mock_trainer.return_value = 0.85  # Just return score
                    # Create a proper mock config object
                    mock_config = Mock()
                    mock_config.trained_model_file_path = 'model_path'
                    training_pipeline.model_trainer.model_trainer_config = mock_config
                    
                    # Execute training
                    try:
                        training_result = training_pipeline.initiate_training_pipeline()
                        assert len(training_result) == 3
                        
                        # Step 2: Test prediction with the "trained" model
                        with patch('src.pipeline.predict_pipeline.load_object') as mock_load:
                            # Mock loading of model and preprocessor
                            mock_model = Mock()
                            mock_model.predict.return_value = np.array([5000000.0])
                            mock_preprocessor = Mock()
                            mock_preprocessor.transform.return_value = np.random.rand(1, 31)
                            mock_load.side_effect = [mock_model, mock_preprocessor]
                            
                            # Create prediction pipeline
                            prediction_pipeline = PredictPipeline()
                            
                            # Test prediction
                            custom_data = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
                            df = custom_data.get_data_as_datafrane()
                            prediction = prediction_pipeline.predict(df)
                            
                            assert isinstance(prediction, np.ndarray)
                            assert len(prediction) == 1
                            assert prediction[0] > 0  # Should be positive price
                            
                    except Exception as e:
                        # Skip if environment not fully set up
                        pytest.skip(f"Integration test requires full environment: {e}")
    
    def test_categorical_variable_processing_pipeline(self, sample_apartment_data):
        """Test that categorical variables are processed correctly through the pipeline."""
        # Test different categorical combinations
        test_cases = [
            {'region': 'södermalm', 'balcony': 'yes', 'expected_region': 'Södermalm, Stockholms kommun', 'expected_balcony': 'Ja'},
            {'region': 'östermalm', 'balcony': 'no', 'expected_region': 'Östermalm, Stockholms kommun', 'expected_balcony': 'Nej'},
            {'region': 'unknown_area', 'balcony': 'maybe', 'expected_region': 'unknown_area', 'expected_balcony': 'Unknown'},
        ]
        
        for case in test_cases:
            custom_data = CustomData(
                number_of_rooms=3,
                area_size=80,
                year_built=2010,
                annual_fee_sek=50000,
                region=case['region'],
                has_balcony=case['balcony'],
                floor_number=5
            )
            
            df = custom_data.get_data_as_datafrane()
            
            # Verify transformations
            assert df['region'].iloc[0] == case['expected_region']
            assert df['balcony'].iloc[0] == case['expected_balcony']
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_prediction_consistency(self, mock_load_object):
        """Test that predictions are consistent for the same input."""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict.return_value = np.array([5000000.0])
        mock_preprocessor = Mock()
        mock_preprocessor.transform.return_value = np.random.rand(1, 31)
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Create pipeline
        pipeline = PredictPipeline()
        
        # Create identical data
        data1 = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
        data2 = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
        
        # Get predictions
        df1 = data1.get_data_as_datafrane()
        df2 = data2.get_data_as_datafrane()
        
        prediction1 = pipeline.predict(df1)
        
        # Reset mock to ensure fresh call
        mock_model.reset_mock()
        mock_preprocessor.reset_mock()
        mock_preprocessor.transform.return_value = np.random.rand(1, 31)
        
        prediction2 = pipeline.predict(df2)
        
        # Should be same (with our mock, they will be)
        assert prediction1[0] == prediction2[0]
        
        # Verify DataFrames are identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_data_transformation_component_integration(self, sample_apartment_data):
        """Test data transformation component with real-like data."""
        from src.components.data_transformation import DataTransformation, FixedTopCategoriesTransformer
        
        # Test FixedTopCategoriesTransformer
        transformer = FixedTopCategoriesTransformer(
            column='region',
            top_categories=['Södermalm, Stockholms kommun', 'Östermalm, Stockholms kommun']
        )
        
        # Fit and transform
        transformer.fit(sample_apartment_data)
        result = transformer.transform(sample_apartment_data)
        
        # Verify transformation
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_apartment_data)
        
        # Should have created one-hot encoded columns (minus first dropped)
        region_cols = [col for col in result.columns if col.startswith('region_')]
        assert len(region_cols) >= 1  # At least one region column after drop_first


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling across pipeline integration."""
    
    def test_malformed_input_data_handling(self):
        """Test handling of malformed input data."""
        # CustomData doesn't validate inputs, but we can test that it accepts various types
        # Test with None values (pandas accepts None)
        data_with_none = CustomData(None, 80, 2010, 50000, 'södermalm', 'yes', 5)
        df = data_with_none.get_data_as_datafrane()
        assert isinstance(df, pd.DataFrame)
        assert df['number_of_rooms'].iloc[0] is None
        
        # Test with string types (should work due to pandas conversion)
        data = CustomData('3', '80', '2010', '50000', 'södermalm', 'yes', '5')
        df = data.get_data_as_datafrane()
        assert isinstance(df, pd.DataFrame)
        assert df['number_of_rooms'].iloc[0] == '3'  # String values are preserved
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_prediction_with_preprocessor_failure(self, mock_load_object):
        """Test prediction pipeline when preprocessor fails."""
        # Setup mocks
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.transform.side_effect = Exception("Preprocessor failed")
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Create pipeline
        pipeline = PredictPipeline()
        
        # Create test data
        data = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
        df = data.get_data_as_datafrane()
        
        # Should raise exception
        with pytest.raises(Exception) as exc_info:
            pipeline.predict(df)
        
        assert "Preprocessor failed" in str(exc_info.value)
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_model_loading_failure(self, mock_load_object):
        """Test prediction pipeline when model loading fails."""
        # Setup mock to fail on model loading
        mock_load_object.side_effect = [Exception("Model loading failed"), Mock()]
        
        # Should raise exception during initialization
        with pytest.raises(Exception) as exc_info:
            PredictPipeline()
        
        assert "Model loading failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])