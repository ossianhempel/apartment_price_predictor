import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
    """
    Complete training pipeline for the apartment price predictor.
    Orchestrates data ingestion, transformation, and model training.
    """
    
    def __init__(self):
        """Initialize the training pipeline components."""
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def initiate_training_pipeline(self):
        """
        Run the complete training pipeline.
        
        Returns:
            tuple: (model_score, model_path, preprocessor_path)
        """
        try:
            logging.info("Starting training pipeline")
            
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logging.info("Step 2: Data Transformation")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Step 3: Model Training
            logging.info("Step 3: Model Training")
            model_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info("Training pipeline completed successfully")
            
            return model_score, self.model_trainer.model_trainer_config.trained_model_file_path, preprocessor_path
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    model_score, model_path, preprocessor_path = training_pipeline.initiate_training_pipeline()
    print(f"Training completed. Model score: {model_score}")
    print(f"Model saved at: {model_path}")
    print(f"Preprocessor saved at: {preprocessor_path}")