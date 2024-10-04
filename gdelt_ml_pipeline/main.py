# main.py

import argparse
from gdelt_ml_pipeline.config import Config
from gdelt_ml_pipeline.data_loader import DataLoader
from gdelt_ml_pipeline.feature_engineering import FeatureEngineering
from gdelt_ml_pipeline.models import StockDataset, init_model
from gdelt_ml_pipeline.train import Trainer
from gdelt_ml_pipeline.evaluate import Evaluator
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import logging
import numpy as np
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    # Load configuration
    config = Config()

    # Data Loading
    data_loader = DataLoader(config)
    hourly_data, stock_data = data_loader.load_data()
    if hourly_data is None or stock_data is None:
        logger.error("Data loading failed.")
        return

    logger.info("Data loaded successfully.")
    logger.info(f"Hourly data: {hourly_data.head()}")
    logger.info(f"Stock data: {stock_data.head()}")

    # Feature Engineering
    feature_engineering = FeatureEngineering(config)
    merged_df = feature_engineering.prepare_dataset(hourly_data, stock_data)

    if len(merged_df) < 2:
        logger.error(
            "Not enough data for modeling. At least 2 rows are required.")
        return

    # Prepare data for modeling
    feature_columns = ['avg_goldstein', 'total_mentions', 'total_sources', 'total_articles', 'avg_tone', 'event_count',
                       'volume_adi', 'momentum_rsi', 'trend_macd'] + list(merged_df.filter(like='x0').columns)
    X = merged_df[feature_columns].values
    y = merged_df['Target'].values

    # Check if we have enough data for train-test split
    if len(X) < 2:
        logger.error(
            "Not enough data for train-test split. At least 2 samples are required.")
        return

    # Initialize model
    model_params = {
        'input_size': X.shape[1],
        'hidden_size': config.hidden_size,
        'num_layers': config.num_layers,
        'output_size': 1,
        'dropout': config.dropout
    }
    model = init_model(config.model_name, model_params)

    # Training and Evaluation
    if config.model_name in ['lstm', 'tft']:
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X, dtype=torch.float32)
        y_train_tensor = torch.tensor(y, dtype=torch.float32)
        X_test_tensor = torch.tensor(X, dtype=torch.float32)
        y_test_tensor = torch.tensor(y, dtype=torch.float32)

        # Create datasets and dataloaders
        train_dataset = StockDataset(X_train_tensor, y_train_tensor)
        test_dataset = StockDataset(X_test_tensor, y_test_tensor)
        train_loader = TorchDataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = TorchDataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False)

        # Training
        trainer = Trainer(config)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate)
        model = trainer.train_lstm(
            model, train_loader, test_loader, criterion, optimizer, config.num_epochs)

        # Evaluation
        evaluator = Evaluator(config)
        evaluator.evaluate_model(model, test_loader)
        evaluator.model_interpretability(model, test_loader)

    elif config.model_name in ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']:
        # For scikit-learn or similar models
        trainer = Trainer(config)
        model = trainer.train_sklearn_model(model, X, y)

        # Evaluation
        evaluator = Evaluator(config)
        evaluator.evaluate_sklearn_model(model, X, y)
        evaluator.model_interpretability_sklearn(model, X)

    else:
        logger.error(f"Model {config.model_name} is not supported.")


if __name__ == "__main__":
    set_seed()
    main()
