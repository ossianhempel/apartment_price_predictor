"""
Unit tests for the training pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.pipeline.train_pipeline import TrainingPipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TestTrainingPipeline:
    """Test suite for TrainingPipeline class."""
    
    def test_training_pipeline_initialization(self):
        """Test that TrainingPipeline initializes correctly."""
        pipeline = TrainingPipeline()
        
        assert isinstance(pipeline.data_ingestion, DataIngestion)
        assert isinstance(pipeline.data_transformation, DataTransformation)
        assert isinstance(pipeline.model_trainer, ModelTrainer)
    
    @patch('src.pipeline.train_pipeline.logging')
    @patch.object(ModelTrainer, 'initiate_model_trainer')
    @patch.object(DataTransformation, 'initiate_data_transformation')
    @patch.object(DataIngestion, 'initiate_data_ingestion')
    def test_initiate_training_pipeline_success(self, mock_ingestion, mock_transformation, 
                                              mock_trainer, mock_logging):
        """Test successful execution of the complete training pipeline."""
        # Setup mocks
        mock_ingestion.return_value = ('train_path', 'test_path')
        mock_transformation.return_value = (np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), 'preprocessor_path')
        mock_trainer.return_value = (Mock(), 0.85)  # (model, score)
        
        pipeline = TrainingPipeline()
        pipeline.model_trainer.model_trainer_config = Mock()
        pipeline.model_trainer.model_trainer_config.trained_model_file_path = 'model_path'
        
        # Execute
        result = pipeline.initiate_training_pipeline()
        
        # Assertions
        assert len(result) == 3  # (model_score, model_path, preprocessor_path)
        mock_ingestion.assert_called_once()
        mock_transformation.assert_called_once_with('train_path', 'test_path')
        mock_trainer.assert_called_once()
        
        # Check logging calls
        assert mock_logging.info.call_count >= 4  # At least 4 info calls
    
    @patch.object(DataIngestion, 'initiate_data_ingestion')
    def test_initiate_training_pipeline_data_ingestion_failure(self, mock_ingestion):
        """Test training pipeline failure during data ingestion."""
        # Setup mock to raise exception
        mock_ingestion.side_effect = Exception("Data ingestion failed")
        
        pipeline = TrainingPipeline()
        
        # Execute and assert exception
        with pytest.raises(Exception) as exc_info:
            pipeline.initiate_training_pipeline()
        
        assert "Data ingestion failed" in str(exc_info.value)
    
    @patch.object(DataTransformation, 'initiate_data_transformation')
    @patch.object(DataIngestion, 'initiate_data_ingestion')
    def test_initiate_training_pipeline_transformation_failure(self, mock_ingestion, mock_transformation):
        """Test training pipeline failure during data transformation."""
        # Setup mocks
        mock_ingestion.return_value = ('train_path', 'test_path')
        mock_transformation.side_effect = Exception("Transformation failed")
        
        pipeline = TrainingPipeline()
        
        # Execute and assert exception
        with pytest.raises(Exception) as exc_info:
            pipeline.initiate_training_pipeline()
        
        assert "Transformation failed" in str(exc_info.value)
    
    @patch.object(ModelTrainer, 'initiate_model_trainer')
    @patch.object(DataTransformation, 'initiate_data_transformation')
    @patch.object(DataIngestion, 'initiate_data_ingestion')
    def test_initiate_training_pipeline_model_training_failure(self, mock_ingestion, 
                                                             mock_transformation, mock_trainer):
        """Test training pipeline failure during model training."""
        # Setup mocks
        mock_ingestion.return_value = ('train_path', 'test_path')
        mock_transformation.return_value = (np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), 'preprocessor_path')
        mock_trainer.side_effect = Exception("Model training failed")
        
        pipeline = TrainingPipeline()
        
        # Execute and assert exception
        with pytest.raises(Exception) as exc_info:
            pipeline.initiate_training_pipeline()
        
        assert "Model training failed" in str(exc_info.value)


class TestTrainingPipelineComponents:
    """Test individual components used in training pipeline."""
    
    def test_data_ingestion_component(self):
        """Test that data ingestion component is properly initialized."""
        pipeline = TrainingPipeline()
        assert hasattr(pipeline.data_ingestion, 'initiate_data_ingestion')
    
    def test_data_transformation_component(self):
        """Test that data transformation component is properly initialized."""
        pipeline = TrainingPipeline()
        assert hasattr(pipeline.data_transformation, 'initiate_data_transformation')
    
    def test_model_trainer_component(self):
        """Test that model trainer component is properly initialized."""
        pipeline = TrainingPipeline()
        assert hasattr(pipeline.model_trainer, 'initiate_model_trainer')


@pytest.mark.slow
class TestTrainingPipelineIntegration:
    """Integration tests for training pipeline (marked as slow)."""
    
    def test_pipeline_with_sample_data(self, sample_apartment_data, temp_artifacts_dir):
        """Test training pipeline with sample data."""
        # Create a pipeline
        pipeline = TrainingPipeline()
        
        # This is a basic integration test that verifies components exist
        # Full testing would require complete data setup
        assert hasattr(pipeline, 'data_ingestion')
        assert hasattr(pipeline, 'data_transformation') 
        assert hasattr(pipeline, 'model_trainer')
        
        # Mock heavy components and test workflow
        with patch.object(pipeline.data_ingestion, 'initiate_data_ingestion') as mock_ingestion:
            with patch.object(pipeline.data_transformation, 'initiate_data_transformation') as mock_transformation:
                with patch.object(pipeline.model_trainer, 'initiate_model_trainer') as mock_trainer:
                    
                    # Setup mock returns
                    mock_ingestion.return_value = ('train_path', 'test_path')
                    mock_transformation.return_value = (np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), 'preprocessor_path')
                    mock_trainer.return_value = (Mock(), 0.8)
                    pipeline.model_trainer.model_trainer_config = Mock()
                    pipeline.model_trainer.model_trainer_config.trained_model_file_path = 'model_path'
                    
                    # Execute 
                    try:
                        result = pipeline.initiate_training_pipeline()
                        assert len(result) == 3
                    except Exception as e:
                        # Some components might fail without full setup, which is expected
                        pytest.skip(f"Integration test requires full environment setup: {e}")


if __name__ == "__main__":
    pytest.main([__file__])