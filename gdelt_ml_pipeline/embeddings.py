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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from openai import RateLimitError
from gdelt_ml_pipeline.config import Config
import os
import numpy as np
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()

client = OpenAI(api_key=config.openai_api_key)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=60),
    before_sleep=before_sleep_log(logger, logging.INFO))
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

    try:
        if embedding_function is None:
            # Default embedding function using OpenAI API
            text = text.replace("\n", " ")
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        else:
            # Use the provided embedding function
            return embedding_function(text)
    except RateLimitError:
        logger.warning("Rate limit exceeded. Retrying...")
        raise
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        return [0] * 1536


def generate_embeddings(
    df: pd.DataFrame,
    embedding_function: Callable = None,
    cache_file: str = None,
    cache_interval: int = 500
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate embeddings for the 'combined' column of a DataFrame with partial caching.
    """
    embeddings = []
    valid_positions = []
    total_rows = len(df)
    logger.info(f"Generating embeddings for {total_rows} rows")

    # Load existing cache if available
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        embeddings = cache['embeddings']
        valid_positions = cache['valid_positions']
        start_position = len(valid_positions)
    else:
        cache = {'embeddings': [], 'valid_positions': []}
        start_position = 0

    for position in tqdm(range(start_position, total_rows), total=total_rows, desc="Generating embeddings", initial=start_position):
        try:
            text = df['combined'].iloc[position]
            embedding = get_embedding(
                text, embedding_function=embedding_function)
            if embedding is not None:
                embeddings.append(embedding)
                valid_positions.append(position)

            # Log progress and cache after every cache_interval rows
            if (position + 1) % cache_interval == 0:
                remaining = total_rows - (position + 1)
                logger.info(
                    f"Processed {position + 1} rows. Remaining: {remaining}")
                if cache_file:
                    cache['embeddings'] = embeddings
                    cache['valid_positions'] = valid_positions
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f)
                    logger.info(f"Cached embeddings up to position {position}")

        except Exception as e:
            logger.error(
                f"Failed to generate embedding for position {position}: {str(e)}")

    # Log final count and cache final results
    logger.info(
        f"Finished processing. Total rows processed: {len(embeddings)}")
    if cache_file:
        cache['embeddings'] = embeddings
        cache['valid_positions'] = valid_positions
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
        logger.info("Cached final embeddings")

    embeddings_array = np.array(embeddings)

    return embeddings_array, valid_positions
