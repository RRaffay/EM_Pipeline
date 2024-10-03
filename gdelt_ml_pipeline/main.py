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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    config = Config()

    # Data Loading
    data_loader = DataLoader(config)
    gdelt_df, stock_data = data_loader.load_data()
    if gdelt_df is None or stock_data is None:
        logger.error("Data loading failed.")
        return

    # Feature Engineering
    feature_engineering = FeatureEngineering(config)
    gdelt_df = feature_engineering.calculate_sentiment(gdelt_df)
    embeddings_df = feature_engineering.aggregate_embeddings(gdelt_df)
    merged_df = feature_engineering.prepare_dataset(embeddings_df, stock_data)

    # Prepare data for modeling
    feature_columns = ['sentiment', 'volume_adi', 'momentum_rsi',
                       'trend_macd'] + list(merged_df.filter(like='x0').columns)
    X = merged_df[feature_columns].values
    y = merged_df['Target'].values
    split_point = int(0.8 * len(y))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train_tensor, y_train_tensor)
    test_dataset = StockDataset(X_test_tensor, y_test_tensor)
    train_loader = TorchDataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = TorchDataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model_params = {
        'input_size': X_train.shape[1],
        'hidden_size': config.hidden_size,
        'num_layers': config.num_layers,
        'output_size': 1,
        'dropout': config.dropout
    }
    model = init_model(config.model_name, model_params)

    # Training
    trainer = Trainer(config)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model = trainer.train_lstm(
        model, train_loader, test_loader, criterion, optimizer, config.num_epochs)

    # Evaluation
    evaluator = Evaluator(config)
    evaluator.evaluate_model(model, test_loader)
    evaluator.model_interpretability(model, test_loader)


if __name__ == "__main__":
    main()
