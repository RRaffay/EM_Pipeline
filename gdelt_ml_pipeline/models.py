# models.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0),
                          self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0),
                          self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class TFTModel(pl.LightningModule):
    def __init__(self, training_data, learning_rate=1e-3):
        super(TFTModel, self).__init__()
        self.model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=learning_rate,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # Quantile regression outputs
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.model.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.model.hparams.learning_rate)
        return optimizer


def init_model(model_name, params):
    if model_name == 'lstm':
        model = LSTMModel(**params)
    elif model_name == 'tft':
        model = TFTModel(**params)
    elif model_name == 'logistic_regression':
        model = LogisticRegression()
    elif model_name == 'random_forest':
        model = RandomForestClassifier()
    elif model_name == 'xgboost':
        model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model
