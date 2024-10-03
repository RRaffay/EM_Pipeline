# feature_engineering.py

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    def __init__(self, config):
        self.config = config

    def aggregate_embeddings(self, gdelt_df):
        """
        Aggregate embeddings by date.
        """
        logger.info("Aggregating embeddings by date.")
        gdelt_df['Date'] = gdelt_df['SQLDATE'].dt.date
        gdelt_df['embedding'] = gdelt_df['embedding'].apply(
            lambda x: np.array(x))
        daily_embeddings = gdelt_df.groupby('Date')['embedding'].apply(
            lambda x: np.mean(np.stack(x), axis=0)).reset_index()
        return daily_embeddings

    def calculate_sentiment(self, gdelt_df):
        """
        Calculate sentiment scores using GDELT's AvgTone.
        """
        logger.info("Calculating sentiment scores.")
        gdelt_df['sentiment'] = gdelt_df['AvgTone'] / \
            100  # Normalize to [-1, 1] range
        return gdelt_df

    def prepare_dataset(self, embeddings_df, stock_data):
        """
        Merge datasets and create features.
        """
        logger.info("Merging embeddings with stock data.")
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
        merged_df = pd.merge(embeddings_df, stock_data[[
                             'Date', 'Close', 'Open', 'High', 'Low', 'Volume']], on='Date', how='inner')
        merged_df.sort_values('Date', inplace=True)

        # Calculate stock returns
        merged_df['Return'] = merged_df['Close'].pct_change()
        merged_df.dropna(subset=['Return'], inplace=True)

        # Create target variable
        merged_df['Target'] = (merged_df['Return'] > 0).astype(int)

        # Add GDELT-specific features
        merged_df = self.add_gdelt_features(merged_df)

        # Add technical indicators
        merged_df = self.add_technical_indicators(merged_df)

        # Add temporal features
        merged_df = self.add_temporal_features(merged_df)

        # Scale features
        merged_df = self.scale_features(merged_df)

        return merged_df

    def add_technical_indicators(self, df):
        """
        Add technical indicators using TA library.
        """
        logger.info("Adding technical indicators.")
        df_ta = df.copy()
        df_ta = add_all_ta_features(
            df_ta, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
        # Select relevant indicators
        technical_columns = ['volume_adi', 'momentum_rsi', 'trend_macd']
        df[technical_columns] = df_ta[technical_columns]
        return df

    def add_temporal_features(self, df):
        """
        Add temporal features like day of week and month.
        """
        logger.info("Adding temporal features.")
        df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['month'] = pd.to_datetime(df['Date']).dt.month
        # One-hot encode temporal features
        encoder = OneHotEncoder(sparse=False)
        temporal_features = encoder.fit_transform(df[['day_of_week', 'month']])
        temporal_feature_names = encoder.get_feature_names_out(
            ['day_of_week', 'month'])
        df_temporal = pd.DataFrame(
            temporal_features, columns=temporal_feature_names)
        df = pd.concat([df.reset_index(drop=True),
                       df_temporal.reset_index(drop=True)], axis=1)
        return df

    def scale_features(self, df):
        """
        Scale numerical features.
        """
        logger.info("Scaling features.")
        scaler = StandardScaler()
        numerical_features = ['sentiment',
                              'volume_adi', 'momentum_rsi', 'trend_macd']
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        return df

    def add_gdelt_features(self, df):
        """
        Add GDELT-specific features.
        """
        logger.info("Adding GDELT-specific features.")
        # Add features based on the new GDELT data structure
        df['event_count'] = df.groupby(
            'Date')['GlobalEventID'].transform('count')
        df['avg_goldstein'] = df.groupby(
            'Date')['GoldsteinScale'].transform('mean')
        df['avg_mentions'] = df.groupby(
            'Date')['NumMentions'].transform('mean')
        df['avg_sources'] = df.groupby('Date')['NumSources'].transform('mean')
        df['avg_articles'] = df.groupby(
            'Date')['NumArticles'].transform('mean')
        return df
