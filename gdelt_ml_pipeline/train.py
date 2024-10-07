# train.py

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import logging
from sklearn.metrics import accuracy_score
from gdelt_ml_pipeline.models import LSTMModel, TFTModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def train_lstm(self, model, train_loader, val_loader, criterion, optimizer, num_epochs):
        model.to(self.device)
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(
                    self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
            # Validation
            accuracy = self.evaluate_lstm(model, val_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), self.config.model_save_path)
            logger.info(
                f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')
        return model

    def evaluate_lstm(self, model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(
                    self.device), y_batch.to(self.device)
                outputs = model(X_batch)
                predicted = (outputs.squeeze() > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        accuracy = correct / total
        return accuracy

    def train_tft(self, model, train_dataloader, val_dataloader):
        trainer = pl.Trainer(max_epochs=self.config.num_epochs,
                             gpus=1 if torch.cuda.is_available() else 0)
        trainer.fit(model, train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)
        return model

    def train_sklearn_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def hyperparameter_tuning(self, train_loader, val_loader):
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'input_size': train_loader.dataset.X.shape[2],
                'hidden_size': trial.suggest_int('hidden_size', 32, 128),
                'num_layers': trial.suggest_int('num_layers', 1, 3),
                'output_size': 1,
                'dropout': trial.suggest_float('dropout', 0.0, 0.5)
            }
            model = LSTMModel(**params).to(self.device)
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=trial.suggest_loguniform('lr', 1e-5, 1e-2))
            # Training loop
            for epoch in range(10):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(
                        self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
                # Validation
                accuracy = self.evaluate_lstm(model, val_loader)
                trial.report(accuracy, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return accuracy
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        logger.info(f'Best trial: {study.best_trial.params}')
        return study.best_trial.params
