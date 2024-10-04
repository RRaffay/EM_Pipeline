# feature_engineering.py

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    def __init__(self, config):
        self.config = config

    def prepare_dataset(self, hourly_data, stock_data):
        """
        Merge datasets and create features.
        """
        logger.info("Merging hourly GDELT data with stock data.")

        # Convert stock_data 'Date' to UTC
        stock_data['Date'] = pd.to_datetime(
            stock_data['Datetime']).dt.tz_convert('UTC').dt.tz_localize(None)

        # Ensure hourly_data 'Date' is in UTC and timezone-naive
        hourly_data['Date'] = pd.to_datetime(hourly_data['Date'])
        if hourly_data['Date'].dt.tz is not None:
            hourly_data['Date'] = hourly_data['Date'].dt.tz_convert(
                'UTC').dt.tz_localize(None)
        else:
            hourly_data['Date'] = hourly_data['Date'].dt.tz_localize(
                'UTC').dt.tz_localize(None)

        merged_df = pd.merge(hourly_data, stock_data[[
                             'Date', 'Close', 'Open', 'High', 'Low', 'Volume']], on='Date', how='inner')
        merged_df.sort_values('Date', inplace=True)

        # Calculate stock returns
        merged_df['Return'] = merged_df['Close'].pct_change()
        merged_df.dropna(subset=['Return'], inplace=True)

        # Create target variable
        merged_df['Target'] = (merged_df['Return'] > 0).astype(int)

        # Add technical indicators
        merged_df = self.add_technical_indicators(merged_df)

        # Add temporal features
        merged_df = self.add_temporal_features(merged_df)

        # Scale features
        merged_df = self.scale_features(merged_df)

        return merged_df

    def add_technical_indicators(self, df):
        """
        Add technical indicators using TA library if there's enough data.
        """
        logger.info("Adding technical indicators.")
        if len(df) > 1:  # Check if there's more than one row of data
            df_ta = df.copy()
            df_ta.set_index('Date', inplace=True)
            df_ta = add_all_ta_features(
                df_ta, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
            # Select relevant indicators
            technical_columns = ['volume_adi', 'momentum_rsi', 'trend_macd']
            df[technical_columns] = df_ta[technical_columns]
        else:
            logger.warning(
                "Not enough data to calculate technical indicators. Skipping this step.")
            # Add placeholder columns with NaN values
            technical_columns = ['volume_adi', 'momentum_rsi', 'trend_macd']
            for col in technical_columns:
                df[col] = np.nan
        return df

    def add_temporal_features(self, df):
        """
        Add temporal features like day of week, month, hour, and minute.
        """
        logger.info("Adding temporal features.")
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['hour'] = df['Date'].dt.hour
        df['minute'] = df['Date'].dt.minute
        # One-hot encode temporal features
        encoder = OneHotEncoder(sparse_output=False)
        temporal_features = encoder.fit_transform(
            df[['day_of_week', 'month', 'hour']])
        temporal_feature_names = encoder.get_feature_names_out(
            ['day_of_week', 'month', 'hour'])
        df_temporal = pd.DataFrame(
            temporal_features, columns=temporal_feature_names)
        df = pd.concat([df.reset_index(drop=True),
                       df_temporal.reset_index(drop=True)], axis=1)
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Scaling features.")
        numerical_features = ['avg_goldstein', 'total_mentions', 'total_sources', 'total_articles', 'avg_tone', 'event_count',
                              'volume_adi', 'momentum_rsi', 'trend_macd']

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        df[numerical_features] = imputer.fit_transform(df[numerical_features])

        # Scale features
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        return df
