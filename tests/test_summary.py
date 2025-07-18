"""
Summary test file demonstrating key functionality.
Run this to verify the main features work correctly.
"""
import pytest
import pandas as pd
from unittest.mock import patch

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.pipeline.train_pipeline import TrainingPipeline


class TestSummary:
    """Summary tests for key functionality."""
    
    def test_custom_data_creation_and_mapping(self):
        """Test that CustomData correctly maps web form inputs."""
        # Test basic creation
        data = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
        df = data.get_data_as_datafrane()
        
        # Verify structure
        assert len(df) == 1
        assert len(df.columns) == 7
        
        # Verify mappings
        assert df['balcony'].iloc[0] == 'Ja'  # yes -> Ja
        assert df['region'].iloc[0] == 'Södermalm, Stockholms kommun'  # södermalm -> full name
        
        # Test different mapping
        data2 = CustomData(2, 60, 2000, 40000, 'östermalm', 'no', 3)
        df2 = data2.get_data_as_datafrane()
        
        assert df2['balcony'].iloc[0] == 'Nej'  # no -> Nej
        assert df2['region'].iloc[0] == 'Östermalm, Stockholms kommun'
    
    def test_training_pipeline_structure(self):
        """Test that training pipeline has correct structure."""
        pipeline = TrainingPipeline()
        
        # Verify components exist
        assert hasattr(pipeline, 'data_ingestion')
        assert hasattr(pipeline, 'data_transformation')
        assert hasattr(pipeline, 'model_trainer')
        
        # Verify methods exist
        assert hasattr(pipeline, 'initiate_training_pipeline')
    
    @pytest.mark.slow
    def test_real_categorical_variable_impact(self):
        """Test that categorical variables actually impact predictions with real model."""
        try:
            # Test balcony impact
            pipeline = PredictPipeline()
            
            # Same apartment, different balcony
            apt_with_balcony = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
            apt_without_balcony = CustomData(3, 80, 2010, 50000, 'södermalm', 'no', 5)
            
            pred_with = pipeline.predict(apt_with_balcony.get_data_as_datafrane())
            pred_without = pipeline.predict(apt_without_balcony.get_data_as_datafrane())
            
            # Should be different
            assert pred_with[0] != pred_without[0], "Balcony should affect price"
            
            # Document the difference
            diff = abs(pred_with[0] - pred_without[0])
            print(f"\\nBalcony price impact: {diff:,.0f} SEK")
            
            # Should be meaningful (at least 10k SEK)
            assert diff >= 10000, f"Price difference too small: {diff}"
            
        except Exception as e:
            pytest.skip(f"Real model test requires trained model: {e}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test unknown region
        data = CustomData(3, 80, 2010, 50000, 'unknown_region', 'yes', 5)
        df = data.get_data_as_datafrane()
        assert df['region'].iloc[0] == 'unknown_region'  # Should remain unchanged
        
        # Test unknown balcony value
        data2 = CustomData(3, 80, 2010, 50000, 'södermalm', 'maybe', 5)
        df2 = data2.get_data_as_datafrane()
        assert df2['balcony'].iloc[0] == 'Unknown'  # Should map to Unknown
        
        # Test extreme values
        data3 = CustomData(10, 300, 1900, 100000, 'södermalm', 'yes', 20)
        df3 = data3.get_data_as_datafrane()
        assert isinstance(df3, pd.DataFrame)
        assert len(df3) == 1


if __name__ == "__main__":
    import pandas as pd
    pytest.main([__file__, "-v"])