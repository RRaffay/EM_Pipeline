# data_loader.py

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
import yfinance as yf
import logging
from gdelt_ml_pipeline.preprocessor import preprocess_data_summary
from gdelt_ml_pipeline.embeddings import generate_embeddings
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config):
        self.config = config
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.google_application_credentials
        self.client = bigquery.Client()

        # Ensure cache directories exist
        os.makedirs(self.config.gdelt_embeddings_cache_dir, exist_ok=True)
        os.makedirs(self.config.gdelt_cache_dir, exist_ok=True)

    def fetch_gdelt_data(self, country_code, hours):
        """
        Fetch GDELT data for the specified country and time range.
        """
        logger.info(
            f"Fetching GDELT data for country: {country_code} for the past {hours} hours.")

        cache_file = os.path.join(
            self.config.gdelt_cache_dir, f"{country_code}_{hours}hours_raw.pkl")

        if self.config.use_cache and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_time, df = pickle.load(f)
            if datetime.now() - cache_time <= self.config.gdelt_cache_expiry:
                logger.info(
                    f"Using cached raw GDELT data from {cache_time}.")
                return df

        query = f"""
            WITH events AS (
                SELECT
                    GlobalEventID,
                    DATEADDED,
                    SQLDATE,
                    Actor1Name,
                    Actor2Name,
                    IsRootEvent,
                    EventCode,
                    EventBaseCode,
                    EventRootCode,
                    QuadClass,
                    GoldsteinScale,
                    NumMentions,
                    NumSources,
                    NumArticles,
                    AvgTone,
                    Actor1Geo_CountryCode,
                    Actor2Geo_CountryCode,
                    ActionGeo_CountryCode,
                    SOURCEURL
                FROM
                    `gdelt-bq.gdeltv2.events_partitioned`
                WHERE
                    _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours + 24} HOUR)
                    AND DATEADDED >= CAST(FORMAT_TIMESTAMP('%Y%m%d%H%M%S', TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)) AS INT64)
                    AND (Actor1Geo_CountryCode = '{country_code}' OR Actor2Geo_CountryCode = '{country_code}' OR ActionGeo_CountryCode = '{country_code}')
            ),
            gkg AS (
                SELECT
                    GKGRECORDID,
                    DATE,
                    SourceCommonName,
                    DocumentIdentifier,
                    V2Themes,
                    V2Locations,
                    V2Persons,
                    V2Organizations,
                    V2Tone,
                    GCAM,
                    AllNames,
                    Amounts,
                    TranslationInfo,
                    Extras
                FROM
                    `gdelt-bq.gdeltv2.gkg_partitioned`
                WHERE
                    _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours + 24} HOUR)
                    AND DATE >= CAST(FORMAT_TIMESTAMP('%Y%m%d%H%M%S', TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)) AS INT64)
            )
            SELECT
                e.*,
                g.GKGRECORDID,
                g.DATE AS GKG_DATE,
                g.SourceCommonName,
                g.V2Themes,
                g.V2Locations,
                g.V2Persons,
                g.V2Organizations,
                g.V2Tone,
                g.GCAM,
                g.AllNames,
                g.Amounts,
                g.TranslationInfo,
                g.Extras
            FROM
                events e
            LEFT JOIN
                gkg g
            ON
                e.SOURCEURL = g.DocumentIdentifier
        """

        try:
            job_config = bigquery.QueryJobConfig(use_query_cache=True)
            query_job = self.client.query(query, job_config=job_config)
            df = query_job.to_dataframe()

            # Convert DATE columns to datetime
            df['SQLDATE'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
            df['GKG_DATE'] = pd.to_datetime(
                df['GKG_DATE'], format='%Y%m%d%H%M%S')
            df['DATEADDED'] = pd.to_datetime(
                df['DATEADDED'], format='%Y%m%d%H%M%S')

            # Remove duplicates
            rows_before = len(df)
            df.drop_duplicates(subset=['SOURCEURL'],
                               keep='first', inplace=True)
            rows_after = len(df)
            logger.info(f"Removed {rows_before - rows_after} duplicate rows.")

            logger.info(f"Fetched {len(df)} rows of GDELT data.")
            if self.config.use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump((datetime.now(), df), f)
                logger.info(f"Cached raw GDELT data.")
            return df
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {e}")
            return pd.DataFrame()

    def preprocess_gdelt_data(self, df):
        """
        Preprocess GDELT data and generate embeddings.
        """
        # Apply preprocessing from preprocessor.py
        df = preprocess_data_summary(df)

        # Generate new embeddings using OpenAI
        embeddings, valid_positions = generate_embeddings(
            df,
            embedding_function=None,  # Use default OpenAI function
        )

        # Add embeddings to the dataframe
        df['embedding'] = pd.Series(
            embeddings.tolist(), index=df.index[valid_positions])

        # Cache the preprocessed data with embeddings
        if self.config.use_cache:
            cache_file = os.path.join(self.config.gdelt_embeddings_cache_dir,
                                      f"{self.config.country_code}_{self.config.hours}hours.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump((datetime.now(), df), f)
            logger.info(f"Cached preprocessed GDELT data with embeddings.")

        return df

    def fetch_stock_data(self, ticker_symbol, start_date, end_date):
        """
        Fetch historical stock data for the specified ticker symbol.
        """
        logger.info(
            f"Fetching stock data for {ticker_symbol} from {start_date} to {end_date}.")
        stock_data = yf.download(ticker_symbol, start=start_date,
                                 end=end_date, interval=self.config.stock_data_interval)
        stock_data.index = stock_data.index.tz_convert('UTC')
        stock_data.reset_index(inplace=True)
        logger.info(f"Fetched {len(stock_data)} rows of stock data.")
        return stock_data

    def aggregate_embeddings(self, gdelt_df):
        """
        Aggregate embeddings and other relevant features by hour.
        """
        logger.info("Aggregating embeddings and features by hour.")
        gdelt_df['Date'] = pd.to_datetime(
            gdelt_df['DATEADDED']).dt.tz_localize('UTC').dt.floor('H')
        gdelt_df['embedding'] = gdelt_df['embedding'].apply(
            lambda x: np.array(x))

        # Aggregate embeddings
        hourly_data = gdelt_df.groupby('Date').agg({
            'embedding': lambda x: np.mean(np.stack(x), axis=0),
            'GoldsteinScale': 'mean',
            'NumMentions': 'sum',
            'NumSources': 'sum',
            'NumArticles': 'sum',
            'AvgTone': 'mean',
            'GlobalEventID': 'count'
        }).reset_index()

        hourly_data.rename(columns={
            'GoldsteinScale': 'avg_goldstein',
            'NumMentions': 'total_mentions',
            'NumSources': 'total_sources',
            'NumArticles': 'total_articles',
            'AvgTone': 'avg_tone',
            'GlobalEventID': 'event_count'
        }, inplace=True)

        return hourly_data

    def load_gdelt_data(self, country_code, hours):
        """
        Load GDELT data, either from cache or by fetching and preprocessing.
        """
        # Check for preprocessed data with embeddings
        preprocessed_cache_file = os.path.join(self.config.gdelt_embeddings_cache_dir, f"{country_code}_{hours}hours_preprocessed.pkl")

        if self.config.use_cache and os.path.exists(preprocessed_cache_file):
            with open(preprocessed_cache_file, 'rb') as f:
                cache_time, df = pickle.load(f)
            if datetime.now() - cache_time <= self.config.embeddings_cache_expiry:
                logger.info(f"Using cached preprocessed GDELT data with embeddings from {cache_time}.")
                return df

        # Fetch raw data (now with its own caching mechanism)
        raw_df = self.fetch_gdelt_data(country_code, hours)
        if raw_df.empty:
            logger.error("No GDELT data fetched.")
            return None

        # Preprocess data and generate embeddings
        try:
            preprocessed_df = self.preprocess_gdelt_data(raw_df)

            # Cache the preprocessed data with embeddings
            if self.config.use_cache:
                with open(preprocessed_cache_file, 'wb') as f:
                    pickle.dump((datetime.now(), preprocessed_df), f)
                logger.info(f"Cached preprocessed GDELT data with embeddings.")

            return preprocessed_df
        except Exception as e:
            logger.error(f"Error preprocessing GDELT data: {e}")
            return None

    def load_data(self):
        """
        Main function to load and preprocess data.
        """
        # Load GDELT data (either from cache or by fetching and preprocessing)
        gdelt_df = self.load_gdelt_data(
            self.config.country_code, self.config.hours)
        if gdelt_df is None:
            return None, None

        # Aggregate GDELT data
        hourly_data = self.aggregate_embeddings(gdelt_df)

        # Fetch stock data
        start_date = hourly_data['Date'].min()
        end_date = hourly_data['Date'].max(
        ) + timedelta(hours=1)  # Include the last hour
        stock_data = self.fetch_stock_data(
            self.config.ticker_symbol, start_date, end_date)
        if stock_data.empty:
            logger.error("No stock data fetched.")
            return None, None

        return hourly_data, stock_data
