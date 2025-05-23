# evaluate.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, test_loader):
        model.eval()
        y_true = []
        y_pred = []
        y_scores = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(next(model.parameters()).device)
                outputs = model(X_batch)
                y_scores.extend(outputs.squeeze().cpu().numpy())
                predicted = (outputs.squeeze() > 0.5).float()
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(y_batch.numpy())
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)
        logger.info(f'Test Accuracy: {accuracy:.4f}')
        logger.info(f'Precision: {precision:.4f}')
        logger.info(f'Recall: {recall:.4f}')
        logger.info(f'ROC AUC Score: {roc_auc:.4f}')
        return accuracy, precision, recall, roc_auc

    def evaluate_sklearn_model(self, model, X_test, y_test):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f'Test Accuracy: {accuracy:.4f}')
        logger.info(f'Precision: {precision:.4f}')
        logger.info(f'Recall: {recall:.4f}')
        logger.info(f'ROC AUC Score: {roc_auc:.4f}')
        return accuracy, precision, recall, roc_auc

    def model_interpretability(self, model, test_loader):
        # Use SHAP for model interpretability (PyTorch models)
        logger.info("Generating SHAP values for model interpretability.")
        X_test = []
        for X_batch, _ in test_loader:
            X_test.append(X_batch.numpy())
        X_test = np.concatenate(X_test, axis=0)
        explainer = shap.DeepExplainer(model, torch.tensor(
            X_test).to(next(model.parameters()).device))
        shap_values = explainer.shap_values(torch.tensor(
            X_test).to(next(model.parameters()).device))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(self.config.shap_summary_plot)

    def model_interpretability_sklearn(self, model, X_test):
        # Use SHAP for model interpretability (scikit-learn models)
        logger.info("Generating SHAP values for model interpretability.")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For binary classification, we typically want the second class's SHAP values
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]

        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(self.config.shap_summary_plot)
        plt.close()
