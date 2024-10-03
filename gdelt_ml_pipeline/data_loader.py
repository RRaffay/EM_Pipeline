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
from gdelt_ml_pipeline.embeddings import generate_embeddings, load_cached_embeddings
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config):
        self.config = config
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.google_application_credentials
        self.client = bigquery.Client()

    def fetch_gdelt_data(self, country_code, hours):
        """
        Fetch GDELT data for the specified country and time range.
        """
        logger.info(
            f"Fetching GDELT data for country: {country_code} for the past {hours} hours.")

        if self.config.use_cache:
            cache_file = os.path.join(
                self.config.gdelt_cache_dir, f"{country_code}_{hours}hours.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_time, df = pickle.load(f)
                if datetime.now() - cache_time <= self.config.gdelt_cache_expiry:
                    logger.info(f"Using cached data from {cache_time}.")
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

            if self.config.use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump((datetime.now(), df), f)

            logger.info(f"Fetched {len(df)} rows of GDELT data.")
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

        # Try to load cached embeddings
        embeddings, valid_positions, cache_valid = load_cached_embeddings(
            self.config)

        if not cache_valid:
            # Generate new embeddings using OpenAI
            embeddings, valid_positions = generate_embeddings(
                df,
                embedding_function=None,  # Use default OpenAI function
                max_workers=self.config.max_workers,
                save_embeddings_path=self.config.save_embeddings_path,
                save_indices_path=self.config.save_indices_path
            )

        # Add embeddings to the dataframe
        df['embedding'] = pd.Series(
            embeddings.tolist(), index=df.index[valid_positions])

        return df

    @lru_cache(maxsize=100)
    def fetch_stock_data(self, ticker_symbol, start_date, end_date):
        """
        Fetch historical stock data for the specified ticker symbol.
        """
        logger.info(
            f"Fetching stock data for {ticker_symbol} from {start_date} to {end_date}.")
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)
        logger.info(f"Fetched {len(stock_data)} rows of stock data.")
        return stock_data

    def load_data(self):
        """
        Main function to load and preprocess data.
        """
        # Fetch GDELT data
        gdelt_df = self.fetch_gdelt_data(
            self.config.country_code, self.config.hours)
        if gdelt_df.empty:
            logger.error("No GDELT data fetched.")
            return None, None

        # Preprocess GDELT data
        gdelt_df = self.preprocess_gdelt_data(gdelt_df)

        # Fetch stock data
        start_date = gdelt_df['SQLDATE'].min().date()
        end_date = gdelt_df['SQLDATE'].max().date(
        ) + timedelta(days=1)  # Include the last day
        stock_data = self.fetch_stock_data(
            self.config.ticker_symbol, start_date, end_date)
        if stock_data.empty:
            logger.error("No stock data fetched.")
            return None, None

        return gdelt_df, stock_data
