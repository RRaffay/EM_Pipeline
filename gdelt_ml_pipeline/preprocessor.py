import pandas as pd
import numpy as np


def safe_get(row, key, default=""):
    """
    Safely get a value from a pandas Series, returning a default if it's empty or null.

    Args:
        row (pandas.Series): The row to get the value from.
        key (str): The key to look up in the row.
        default (Any, optional): The default value to return if the key is not found or the value is empty/null. Defaults to "".

    Returns:
        Any: The value associated with the key, or the default value if not found or empty/null.
    """
    value = row.get(key, default)
    if pd.isna(value) or value == "":
        return default
    return value


def interpret_avg_tone(avg_tone: float) -> str:
    """
    Interpret the average tone of an event and return a descriptive string.

    Args:
        avg_tone (float): The average tone value to interpret.

    Returns:
        str: A string describing the tone of the event based on the avg_tone value.
    """
    if pd.isna(avg_tone):
        return "an unknown tone"
    if avg_tone > 5:
        return "a very positive tone"
    elif avg_tone > 0:
        return "a somewhat positive tone"
    elif avg_tone == 0:
        return "a neutral tone"
    elif avg_tone > -5:
        return "a somewhat negative tone"
    else:
        return "a very negative tone"


def interpret_goldstein_scale(scale: float) -> str:
    """
    Interpret the Goldstein scale value of an event and return a descriptive string.

    Args:
        scale (float): The Goldstein scale value to interpret.

    Returns:
        str: A string describing the impact of the event based on the Goldstein scale value.
    """
    if pd.isna(scale):
        return "an event with unknown impact"
    if scale > 7:
        return "a highly cooperative event"
    elif scale > 3:
        return "a moderately cooperative event"
    elif scale > -3:
        return "a neutral to slightly impactful event"
    elif scale > -7:
        return "a moderately conflictual event"
    else:
        return "a highly conflictual event"


def interpret_quad_class(quad_class: int) -> str:
    """
    Interpret the quad class of an event and return a descriptive string.

    Args:
        quad_class (int): The quad class value to interpret.

    Returns:
        str: A string describing the classification of the event based on the quad class value.
    """
    quad_class_dict = {
        1: "Verbal Cooperation",
        2: "Material Cooperation",
        3: "Verbal Conflict",
        4: "Material Conflict"
    }
    if pd.isna(quad_class):
        return "Unknown classification"
    return quad_class_dict.get(int(quad_class), "Unknown classification")


def preprocess(entry: str, column_type: str, max_entities: int = 10) -> str:
    """
    Preprocesses the given entry based on the specified column type.
    Returns up to 10 unique entities.
    """
    if not isinstance(entry, str) or pd.isna(entry):
        return ""
    mentions = entry.split(";")
    if column_type == "location":
        names = [mention.split("#")[2]
                 for mention in mentions if len(mention.split("#")) > 2]
    else:
        names = [mention.split(",")[0].replace(" ", "_")
                 for mention in mentions]
    # Remove duplicates and non-informative entities
    unique_names = list(dict.fromkeys(names))
    return ", ".join(unique_names[:max_entities])


def preprocess_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the processed columns into a summary column that is used to create the embeddings.
    Includes additional event information for better context and handles empty values.
    """
    required_columns = [
        "V2Persons", "V2Organizations", "V2Locations", "V2Themes",
        "SQLDATE", "EventCode", "AvgTone", "GoldsteinScale", "QuadClass",
        "Actor1Name", "Actor2Name", "NumMentions", "NumSources", "NumArticles"
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    max_entities = {
        "person": 10,
        "organization": 10,
        "location": 10,
        "theme": 4
    }

    df['processed_persons'] = df['V2Persons'].apply(
        lambda x: preprocess(x, "person", max_entities=max_entities["person"]))
    df['processed_organizations'] = df['V2Organizations'].apply(
        lambda x: preprocess(x, "organization", max_entities=max_entities["organization"]))
    df['processed_locations'] = df['V2Locations'].apply(
        lambda x: preprocess(x, "location", max_entities=max_entities["location"]))
    df['processed_themes'] = df['V2Themes'].apply(
        lambda x: preprocess(x, "theme", max_entities=max_entities["theme"]))

    # df['combined'] = df.apply(
    #     lambda row: (
    #         f"On {safe_get(row, 'SQLDATE', 'an unknown date')}, an event occurred with the following details. "
    #         f"It has the CAMEO code {safe_get(row, 'EventCode', 'unknown')}. "
    #         f"The event is classified as {interpret_quad_class(safe_get(row, 'QuadClass'))}. "
    #         f"The Goldstein scale value is {safe_get(row, 'GoldsteinScale', 'unknown')}, indicating {interpret_goldstein_scale(safe_get(row, 'GoldsteinScale'))}. "
    #         f"The average tone was {safe_get(row, 'AvgTone', 'unknown')}, suggesting {interpret_avg_tone(safe_get(row, 'AvgTone'))}. "
    #         f"The primary actors involved were {safe_get(row, 'Actor1Name', 'unknown')} and {safe_get(row, 'Actor2Name', 'unknown')}. "
    #         f"This event was mentioned {safe_get(row, 'NumMentions', 'an unknown number of')} times across {safe_get(row, 'NumSources', 'an unknown number of')} sources in {safe_get(row, 'NumArticles', 'an unknown number of')} articles. "
    #         f"Involved persons: {row['processed_persons'] or 'None mentioned'}. "
    #         f"Involved organizations: {row['processed_organizations'] or 'None mentioned'}. "
    #         f"Locations: {row['processed_locations'] or 'None mentioned'}. "
    #         f"Themes associated: {row['processed_themes'] or 'None mentioned'}."
    #     ),
    #     axis=1
    # )

    df['combined'] = df.apply(
        lambda row: (
            f"On {safe_get(row, 'SQLDATE', 'an unknown date')}, an event occurred with the following details. "
            f"Involved persons: {row['processed_persons'] or 'None mentioned'}. "
            f"Involved organizations: {row['processed_organizations'] or 'None mentioned'}. "
            f"Locations: {row['processed_locations'] or 'None mentioned'}. "
            f"Themes associated: {row['processed_themes'] or 'None mentioned'}."
        ),
        axis=1
    )

    return df


def enrich_user_interest(input_sentence: str) -> str:
    """
    Enrich the user's area of interest by analyzing and expanding the input sentence.

    This function should:
    - Analyze the input sentence to extract key topics and entities.
    - Expand the query using synonyms, related terms, or domain-specific knowledge.
    - Return an enriched query or set of keywords to improve cluster matching.

    Args:
        input_sentence (str): The user's input sentence describing their area of interest.

    Returns:
        str: An enriched query or set of keywords based on the input sentence.

    Note:
        This is currently a placeholder method. The actual implementation should include
        NLP methods, query expansion techniques, and possibly integration with external
        knowledge bases or thesauri.
    """
    # TODO: Implement query enrichment using techniques like:
    # - NLP methods to extract entities and keywords.
    # - Query expansion using a knowledge base or thesaurus.
    # - Synonym expansion using WordNet or similar resources.
    # For now, return the input sentence as-is.
    return input_sentence
