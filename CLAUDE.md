# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning apartment price predictor that:

- Trains models on scraped apartment listing data using CatBoost and other ML libraries
- Serves predictions via a Flask web application
- Uses a modular ML pipeline architecture with separate data ingestion, transformation, validation, training, and prediction components

## Common Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run Flask web application (development with auto-reload)
python dev_server.py
# Enhanced development server with file watching
# Access at http://127.0.0.1:3000

# Alternative: Basic Flask development server
python app.py
# Access at http://127.0.0.1:3000

# Run Flask web application (production)
python main.py
```

### Training Pipeline
```bash
# Run full ML training pipeline
python src/pipeline/train_pipeline.py
```

### Testing
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests  
pytest tests/integration/

# Run tests excluding slow tests
pytest -m "not slow"

# Run specific test file
pytest tests/unit/test_predict_pipeline.py

# Run tests with verbose output
pytest -v
```

## Architecture

### Core Components (`src/components/`)

- `data_ingestion.py` - Handles loading and splitting data
- `data_transformation.py` - Feature engineering and preprocessing 
- `data_validation.py` - Data quality checks
- `model_trainer.py` - Model training with hyperparameter tuning
- `model_evaluation.py` - Model performance evaluation

### Pipeline (`src/pipeline/`)

- `train_pipeline.py` - Orchestrates full training workflow
- `predict_pipeline.py` - Contains `PredictPipeline` class for inference and `CustomData` class for input mapping

### Artifacts (`artifacts/`)

- `model.pkl` - Latest trained model (various dated versions available)
- `preprocessor.pkl` - Fitted preprocessing pipeline
- Training/test data splits

### Web Application

- `app.py` - Flask app with prediction endpoint at root (`/`)
- `templates/prediction_form.html` - Web form for apartment features
- Form inputs: number_of_rooms, area_size, year_built, annual_fee_sek, region, has_balcony, floor_number
- Returns prediction in millions SEK

### Utilities

- `src/utils.py` - Common utilities for loading/saving objects
- `src/logger.py` - Logging configuration
- `src/exception.py` - Custom exception handling

## Data Flow

1. Training: Raw data → Data Ingestion → Transformation → Validation → Model Training → Artifacts
2. Prediction: Web form → CustomData → PredictPipeline → Preprocessor → Model → Result

## Deployment

### Docker Registry
- Production image: `ossianhempel/apartment-price-predictor`
- Deployed on VM using Coolify
- Update deployment by pushing new image to registry

```bash
# Build and push to registry
docker build -t ossianhempel/apartment-price-predictor .
docker push ossianhempel/apartment-price-predictor

# Local development
docker run -p 3000:3000 ossianhempel/apartment-price-predictor
```

The Dockerfile:

- Uses Python 3.10 base image
- Exposes port 3000
- Runs `main.py` (production Flask app) by default

### CI/CD

- Basic GitHub Actions workflow exists (`.github/workflows/github-actions-demo.yml`)
- Currently just a demo workflow - needs proper CI/CD pipeline setup

## Development Notes

- Models are serialized using pickle and stored in `artifacts/`
- The prediction pipeline loads the latest model (`model_2024_03_11.pkl`) and preprocessor
- No test files currently exist in the codebase
- Jupyter notebooks in `notebook/` contain data exploration and model development work
