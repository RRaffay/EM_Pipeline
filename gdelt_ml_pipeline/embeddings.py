from typing import Callable
import time
import concurrent.futures
from typing import List, Tuple, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from openai import OpenAI
from functools import partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from openai import OpenAI
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from gdelt_ml_pipeline.config import Config
import os
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()

client = OpenAI(api_key=config.openai_api_key)


# @lru_cache(maxsize=1000)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
def get_embedding(text: str, model: str = "text-embedding-3-small", embedding_function: Callable = None) -> List[float]:
    """
    Generate an embedding for a given text using the specified embedding function or OpenAI's API.

    Args:
        text (str): The input text to generate an embedding for.
        model (str, optional): The OpenAI model to use for embedding. Defaults to "text-embedding-3-small".
        embedding_function (Callable, optional): A custom embedding function to use. If None, uses OpenAI's API. Defaults to None.

    Returns:
        List[float]: The generated embedding as a list of floats.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError("Input text cannot be empty")

    if embedding_function is None:
        # Default embedding function using OpenAI API
        try:
            text = text.replace("\n", " ")
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None
    else:
        # Use the provided embedding function
        return embedding_function(text)


def generate_embeddings(
    df: pd.DataFrame,
    embedding_function: Callable = None,
    max_workers: int = 3,
    save_embeddings_path: str = None,
    save_indices_path: str = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate embeddings for the 'combined' column of a DataFrame using parallel processing.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'combined' column with text to embed.
        embedding_function (Callable, optional): A custom embedding function to use. If None, uses OpenAI's API. Defaults to None.
        max_workers (int, optional): The maximum number of worker threads to use for parallel processing. Defaults to 3.
        save_embeddings_path (str, optional): File path to save the generated embeddings. If None, embeddings are not saved. Defaults to None.
        save_indices_path (str, optional): File path to save the valid position indices. If None, indices are not saved. Defaults to None.

    Returns:
        Tuple[np.ndarray, List[int]]: A tuple containing:
            - np.ndarray: The generated embeddings as a 2D numpy array.
            - List[int]: The list of valid position indices corresponding to successful embeddings.
    """
    embeddings = []
    valid_positions = []

    def process_row(position, text):
        """
        Process a single row to generate an embedding.

        Args:
            position (int): The position (index) of the row in the DataFrame.
            text (str): The text to generate an embedding for.

        Returns:
            Tuple[int, List[float]] or None: A tuple containing the position and the generated embedding,
            or None if the embedding generation fails.
        """
        try:
            embedding = get_embedding(
                text, embedding_function=embedding_function)
            if embedding is not None:
                return position, embedding
            return None
        except Exception as e:
            logger.error(
                f"Failed to generate embedding for position {position}: {str(e)}")
            return None

    total_rows = len(df)
    positions = range(total_rows)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(process_row, positions, df['combined']), total=total_rows))
    for result in results:
        if result is not None:
            position, embedding = result
            embeddings.append(embedding)
            valid_positions.append(position)

    embeddings_array = np.array(embeddings)

    # Save embeddings to file if path is provided
    if save_embeddings_path is not None:
        np.save(save_embeddings_path, embeddings_array)
        logger.info(f"Embeddings saved to {save_embeddings_path}")

    if save_indices_path is not None:
        np.save(save_indices_path, np.array(valid_positions))
        logger.info(f"Valid positions saved to {save_indices_path}")

    return embeddings_array, valid_positions


def load_cached_embeddings(config):
    """
    Load cached embeddings if they exist and are not expired.

    Args:
        config (Config): The configuration object containing cache paths and expiry time.

    Returns:
        Tuple[np.ndarray, List[int], bool]: A tuple containing:
            - np.ndarray: The loaded embeddings as a 2D numpy array, or None if not found or expired.
            - List[int]: The list of valid position indices, or None if not found or expired.
            - bool: True if valid cached embeddings were loaded, False otherwise.
    """
    if os.path.exists(config.save_embeddings_path) and os.path.exists(config.save_indices_path):
        embeddings_mtime = os.path.getmtime(config.save_embeddings_path)
        indices_mtime = os.path.getmtime(config.save_indices_path)
        cache_time = datetime.fromtimestamp(
            min(embeddings_mtime, indices_mtime))

        if datetime.now() - cache_time <= config.embeddings_cache_expiry:
            embeddings = np.load(config.save_embeddings_path)
            valid_positions = np.load(config.save_indices_path).tolist()
            logger.info(f"Loaded cached embeddings from {cache_time}")
            return embeddings, valid_positions, True

    logger.info("No valid cached embeddings found")
    return None, None, False
