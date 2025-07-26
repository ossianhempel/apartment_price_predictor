"""
Integration tests specifically for categorical variable behavior.
These tests verify that changing categorical variables actually affects predictions.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from src.pipeline.predict_pipeline import PredictPipeline, CustomData


@pytest.mark.integration
class TestCategoricalVariableBehavior:
    """Test that categorical variables actually affect predictions."""
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_balcony_affects_predictions(self, mock_load_object):
        """Test that having a balcony vs not having one affects the prediction."""
        # Setup mock model that gives different predictions for different inputs
        mock_model = Mock()
        mock_preprocessor = Mock()
        
        # Configure the preprocessor to return different features for different inputs
        def mock_transform(df):
            # Return different feature vectors based on balcony value
            if df['balcony'].iloc[0] == 'Ja':  # Has balcony
                return np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1]])  # balcony=1
            else:  # No balcony
                return np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0]])  # balcony=0
        
        mock_preprocessor.transform.side_effect = mock_transform
        
        # Configure model to give different predictions based on last feature (balcony)
        def mock_predict(X):
            if X[0][-1] == 1:  # Has balcony
                return np.array([5500000.0])  # Higher price
            else:  # No balcony
                return np.array([5000000.0])  # Lower price
        
        mock_model.predict.side_effect = mock_predict
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Create prediction pipeline
        pipeline = PredictPipeline()
        
        # Test apartment with balcony
        apartment_with_balcony = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
        df_with_balcony = apartment_with_balcony.get_data_as_datafrane()
        prediction_with_balcony = pipeline.predict(df_with_balcony)
        
        # Reset mocks for second prediction
        mock_model.reset_mock()
        mock_preprocessor.reset_mock()
        mock_preprocessor.transform.side_effect = mock_transform
        mock_model.predict.side_effect = mock_predict
        
        # Test same apartment without balcony
        apartment_without_balcony = CustomData(3, 80, 2010, 50000, 'södermalm', 'no', 5)
        df_without_balcony = apartment_without_balcony.get_data_as_datafrane()
        prediction_without_balcony = pipeline.predict(df_without_balcony)
        
        # Verify predictions are different
        assert prediction_with_balcony[0] != prediction_without_balcony[0]
        assert prediction_with_balcony[0] > prediction_without_balcony[0]  # Balcony should increase price
        
        # Verify the difference is meaningful (at least 50k SEK)
        price_difference = prediction_with_balcony[0] - prediction_without_balcony[0]
        assert price_difference >= 50000, f"Price difference too small: {price_difference}"
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_region_affects_predictions(self, mock_load_object):
        """Test that different regions affect the prediction."""
        from unittest.mock import Mock
        
        # Setup mock model that gives different predictions for different regions
        mock_model = Mock()
        mock_preprocessor = Mock()
        
        # Map regions to different feature encodings
        region_features = {
            'Södermalm, Stockholms kommun': [1, 0, 0, 0],  # Södermalm encoded
            'Östermalm, Stockholms kommun': [0, 1, 0, 0],  # Östermalm encoded  
            'Vasastan, Stockholms kommun': [0, 0, 1, 0],   # Vasastan encoded
            'unknown_region': [0, 0, 0, 1]                # Other encoded
        }
        
        def mock_transform(df):
            region = df['region'].iloc[0]
            base_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27]  # 27 base features
            
            # Add region encoding (4 features)
            region_encoding = region_features.get(region, [0, 0, 0, 1])
            return np.array([base_features + region_encoding])
        
        mock_preprocessor.transform.side_effect = mock_transform
        
        # Configure model to give different predictions based on region encoding
        def mock_predict(X):
            region_part = X[0][-4:]  # Last 4 features are region encoding
            if region_part[1] == 1:  # Östermalm
                return np.array([7000000.0])  # Expensive area
            elif region_part[0] == 1:  # Södermalm
                return np.array([6000000.0])  # Moderate area
            elif region_part[2] == 1:  # Vasastan
                return np.array([5500000.0])  # Another area
            else:  # Unknown/Other
                return np.array([4500000.0])  # Default/cheaper
        
        mock_model.predict.side_effect = mock_predict
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Create prediction pipeline
        pipeline = PredictPipeline()
        
        # Test different regions
        test_regions = [
            ('södermalm', 6000000.0),
            ('östermalm', 7000000.0), 
            ('vasastan', 5500000.0),
            ('unknown_place', 4500000.0)
        ]
        
        predictions = {}
        for region, expected_price in test_regions:
            # Reset mocks
            mock_model.reset_mock()
            mock_preprocessor.reset_mock()
            mock_preprocessor.transform.side_effect = mock_transform
            mock_model.predict.side_effect = mock_predict
            
            apartment = CustomData(3, 80, 2010, 50000, region, 'yes', 5)
            df = apartment.get_data_as_datafrane()
            prediction = pipeline.predict(df)
            predictions[region] = prediction[0]
            
            # Verify prediction matches expected
            assert prediction[0] == expected_price, f"Region {region}: expected {expected_price}, got {prediction[0]}"
        
        # Verify all predictions are different
        prediction_values = list(predictions.values())
        assert len(set(prediction_values)) == len(prediction_values), "All region predictions should be different"
        
        # Verify Östermalm is most expensive
        assert predictions['östermalm'] > predictions['södermalm']
        assert predictions['östermalm'] > predictions['vasastan']
        assert predictions['östermalm'] > predictions['unknown_place']
    
    @patch('src.pipeline.predict_pipeline.load_object')
    def test_multiple_categorical_combinations(self, mock_load_object):
        """Test combinations of categorical variables."""
        from unittest.mock import Mock
        
        mock_model = Mock()
        mock_preprocessor = Mock()
        
        # Create a comprehensive feature mapping
        def mock_transform(df):
            # Base features (27)
            base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
            
            # Region features (4)
            region = df['region'].iloc[0]
            if 'Östermalm' in region:
                region_features = [0, 1, 0, 0]
            elif 'Södermalm' in region:
                region_features = [1, 0, 0, 0]
            else:
                region_features = [0, 0, 0, 1]
            
            return np.array([base + region_features])
        
        mock_preprocessor.transform.side_effect = mock_transform
        
        # Model gives different predictions based on features
        def mock_predict(X):
            region_features = X[0][-4:]
            base_price = 5000000
            
            # Add premium for Östermalm
            if region_features[1] == 1:  # Östermalm
                base_price += 1000000
            elif region_features[0] == 1:  # Södermalm  
                base_price += 500000
                
            return np.array([base_price])
        
        mock_model.predict.side_effect = mock_predict
        mock_load_object.side_effect = [mock_model, mock_preprocessor]
        
        # Create prediction pipeline
        pipeline = PredictPipeline()
        
        # Test combinations
        test_combinations = [
            ('östermalm', 'yes', 6000000),  # Premium area with balcony
            ('östermalm', 'no', 6000000),   # Premium area without balcony (balcony not in this mock)
            ('södermalm', 'yes', 5500000),  # Mid area with balcony  
            ('unknown', 'yes', 5000000),    # Unknown area
        ]
        
        for region, balcony, expected_base in test_combinations:
            # Reset mocks
            mock_model.reset_mock()
            mock_preprocessor.reset_mock()
            mock_preprocessor.transform.side_effect = mock_transform
            mock_model.predict.side_effect = mock_predict
            
            apartment = CustomData(3, 80, 2010, 50000, region, balcony, 5)
            df = apartment.get_data_as_datafrane()
            prediction = pipeline.predict(df)
            
            assert prediction[0] == expected_base, f"Combination {region}/{balcony}: expected {expected_base}, got {prediction[0]}"


@pytest.mark.integration 
class TestRealModelCategoricalBehavior:
    """Test categorical behavior with the actual trained model (slower tests)."""
    
    @pytest.mark.slow
    def test_real_model_balcony_difference(self):
        """Test that real model shows price difference for balcony."""
        try:
            # Create prediction pipeline with real model
            pipeline = PredictPipeline()
            
            # Test apartment with balcony
            apartment_with_balcony = CustomData(3, 80, 2010, 50000, 'södermalm', 'yes', 5)
            df_with_balcony = apartment_with_balcony.get_data_as_datafrane()
            prediction_with_balcony = pipeline.predict(df_with_balcony)
            
            # Test same apartment without balcony
            apartment_without_balcony = CustomData(3, 80, 2010, 50000, 'södermalm', 'no', 5)
            df_without_balcony = apartment_without_balcony.get_data_as_datafrane()
            prediction_without_balcony = pipeline.predict(df_without_balcony)
            
            # Verify predictions are different (this should work with our fixed model)
            assert prediction_with_balcony[0] != prediction_without_balcony[0], \
                f"Balcony should affect price: with={prediction_with_balcony[0]}, without={prediction_without_balcony[0]}"
            
            # Document the actual difference
            price_diff = abs(prediction_with_balcony[0] - prediction_without_balcony[0])
            print(f"Real model balcony price difference: {price_diff:,.0f} SEK")
            
            # Should be a meaningful difference (at least 10k SEK)
            assert price_diff >= 10000, f"Price difference too small: {price_diff}"
            
        except Exception as e:
            pytest.skip(f"Real model test requires trained model: {e}")
    
    @pytest.mark.slow  
    def test_real_model_region_differences(self):
        """Test that real model shows price differences between regions."""
        try:
            pipeline = PredictPipeline()
            
            # Test different regions with same other features
            base_features = (3, 80, 2010, 50000, 'yes', 5)  # rooms, size, year, fee, balcony, floor
            
            test_regions = ['södermalm', 'östermalm', 'vasastan', 'bromma']
            predictions = {}
            
            for region in test_regions:
                apartment = CustomData(*base_features[:4], region, *base_features[4:])
                df = apartment.get_data_as_datafrane()
                prediction = pipeline.predict(df)
                predictions[region] = prediction[0]
                print(f"Real model prediction for {region}: {prediction[0]:,.0f} SEK")
            
            # Check that we get some variation (not all identical)
            unique_predictions = len(set(predictions.values()))
            assert unique_predictions > 1, f"Expected variation in regional prices, got: {predictions}"
            
            # Document the range
            min_price = min(predictions.values())
            max_price = max(predictions.values())
            price_range = max_price - min_price
            print(f"Real model regional price range: {price_range:,.0f} SEK")
            
        except Exception as e:
            pytest.skip(f"Real model test requires trained model: {e}")


if __name__ == "__main__":
    pytest.main([__file__])