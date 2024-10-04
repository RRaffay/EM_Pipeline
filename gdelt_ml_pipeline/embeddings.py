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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError
from gdelt_ml_pipeline.config import Config
import os
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()

client = OpenAI(api_key=config.openai_api_key)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=60))
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
    max_workers: int = 3
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate embeddings for the 'combined' column of a DataFrame using parallel processing.
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

    return embeddings_array, valid_positions
