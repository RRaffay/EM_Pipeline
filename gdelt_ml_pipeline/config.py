# config.py
from datetime import timedelta
import os
from dotenv import load_dotenv


class Config:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Data parameters
        self.country_code = 'IN'  # India
        self.ticker_symbol = '^NSEI'  # Nifty 50 Index
        self.hours = 120  # 5 days (24 hours * 5)
        self.user_interest = "Economic growth and technology sector in India"

        # Caching parameters
        self.use_cache = True
        self.gdelt_cache_dir = "cache/gdelt_cache"
        self.gdelt_cache_expiry = timedelta(hours=24)

        # Embedding parameters
        self.max_workers = 5
        self.save_embeddings_path = "cache/embeddings_cache/embeddings.npy"
        self.save_indices_path = "cache/embeddings_cache/indices.npy"
        self.embeddings_cache_expiry = timedelta(
            days=7)  # Cache embeddings for 7 days

        # Model parameters
        self.model_name = 'lstm'  # 'lstm' or 'tft'
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.2
        self.learning_rate = 1e-3
        self.num_epochs = 30
        self.batch_size = 32
        self.model_save_path = 'best_model.pth'

        # Evaluation parameters
        self.shap_summary_plot = 'shap_summary.png'

        # Google Cloud credentials (loaded from .env)
        self.google_application_credentials = os.getenv(
            'GOOGLE_APPLICATION_CREDENTIALS', 'config/em-news-gdelt.json')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
