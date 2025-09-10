"""
Helper functions for the taxonomic RAG system.

Together, these functions streamline the workflow for
taxonomic classification tasks, enabling efficient data processing, evaluation,
and reporting in the taxonomic RAG system.

This module provides a comprehensive set of utility functions to facilitate
various tasks in the taxonomic RAG system. These include document formatting
for LLM prompts, image processing for handling base64 and PIL image conversions,
and string formatting for generating detailed taxonomic classification outputs.
Additionally, the module supports classification evaluation by computing metrics
such as accuracy and F1 scores, comparing true to predicted classifications,
and generating classification reports of taxonomic predictions.

Also includes data handling utilities for processing batches of
data, RAG evaluation for assessing response quality using faithfulness and
relevancy metrics, and file writing utilities for exporting metrics and
predictions to CSV files.

Dependencies:
    - base64
    - csv
    - numpy
    - pandas
    - requests
    - langchain
    - langchain_openai
    - PIL (Pillow)
    - ragas
    - sklearn

Functions:
    - format_context: Format documents for pretty and informative output.
    - format_docs: Format context for input into an LLM prompt.
    - unique_docs: Get unique retrieved document chunks from multiple queries.
    - simple_string_output: Generate a formatted string for taxonomic classification.
    - clean_string_output: Format and return a string representation of observations.
    - b64_to_pil: Convert a base64-encoded image to a PIL Image object.
    - imgurl_tob64: Convert an image from a URL to a Base64-encoded string.
    - imgfile_tob64: Convert an image file to a Base64-encoded string.
    - pilimg_tob64: Convert a PIL Image object to a Base64-encoded string.
    - get_metrics: Calculate accuracy and F1 score metrics for labels.
    - dict_match: Compare dictionaries to determine matching key-value pairs.
    - classify_report: Generate classification metrics for taxonomic predictions.
    - custom_collate_fn: Process batches of data using a custom collate function.
    - rag_evaluate: Evaluate response quality in a RAG system.
    - write_overall_metrics: Write overall metrics to a CSV file.
    - extract_tax_metrics: Extract classification metrics from a result object.
    - extract_tax_metrics_rs: Extract metrics and enrich guess class dictionaries.
    - write_preds_to_csv: Write prediction data to a CSV file.
"""

import base64
import csv
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from langchain.load import dumps, loads
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from PIL import Image
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import json

RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def load_api_keys() -> None:
    """Load API keys from files.

    OpenAI API key is required, OpenRouter and Cohere API keys are optional.
    Read API keys from files in home dir - `~/.openai.key`,
    `~/.openrouter.key` and `~/.cohere.key`.
    """
    try:
        with open(Path.home() / ".openai.key", "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find OpenAI API key at ~/.openai.key. "
            "This key is required for the model to function."
        ) from None

    try:
        with open(Path.home() / ".openrouter.key", "r") as f:
            os.environ["OPENROUTER_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        # OpenRouter key is optional, only log a warning
        print(
            "Warning: Could not find OpenRouter API key at ~/.openrouter.key. "
            "This is fine if you're not using OpenRouter models."
        )

    try:
        with open(Path.home() / ".cohere.key", "r") as f:
            os.environ["COHERE_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        # Cohere key is optional, only log a warning
        print(
            "Warning: Could not find Cohere API key at ~/.cohere.key. "
            "This is fine if you're not using reranking functionality."
        )


def format_context(docs: List[Document]) -> str:
    """
    Format documents for pretty and informative output.

    Will return a single string with source information and relevancy scores.

    :param docs: Retrieved document chunks
    :return: Single string of all document chunks with source information
             and relevancy scores
    """
    out = []
    for i, d in enumerate(docs):
        # Create source string (include page if part of multi-page document)
        if d.metadata.get("page"):
            source_line = f"\tsource: page {d.metadata['page']} of {d.metadata['source'].split('/')[-1]}"
        else:
            source_line = f"\tsource: {d.metadata['source'].split('/')[-1]}"
        if d.metadata.get("relevance_score") is None:
            d.metadata["relevance_score"] = "N/A"
        # Construct output string with source, relevancy score and actual content
        out.append(
            f"""Document {i + 1}:
        {source_line}
        \trelevance: {d.metadata["relevance_score"]}
        \n{d.page_content}\n\n"""
        )
    # Merge each into single output string
    return f"\n{'-' * 100}\n".join(out)


def format_docs(docs: List[Document]) -> str:
    """
    Format context for input into an LLM prompt.

    :param docs: Retrieved document chunks
    :return: Single string containing all fetched document chunks
    """
    return "\n\n".join(f"{str(doc.metadata)}\n{doc.page_content}" for doc in docs)


def unique_docs(docs: List[List[Document]]) -> List[Document]:
    """
    Get unique retrieved document chunks from list of several retriever queries.

    :param docs: All retrieved document chunks (from several retrievals) with duplicates
    :return: Unique document chunks
    """
    # Flatten list of lists and convert each to string
    flattened_docs = [dumps(doc) for sublist in docs for doc in sublist]
    unique_chunks = list(set(flattened_docs))  # Keep only unique chunks
    return [loads(doc) for doc in unique_chunks]


def simple_string_output(out_dict: Dict[str, Union[str, Dict[str, str]]]) -> str:
    """Generate a formatted string of taxonomic classification results.

    Args:
        out_dict (dict): A dictionary of output from the RAG models containing the keys:
            - "guess_class": The taxonomic classification guess.
            - "ancestral": Information about ancestral features.
            - "specific": Information about organismal features.
            - "biodiversity": Knowledge about biodiversity.
            - "commentary": Additional commentary.

    Returns
    -------
        str: A formatted string containing the model's structured outputs.
    """
    return f"""
Taxonomic Classification:
{out_dict["guess_class"]}
Ancestral features:
{out_dict["ancestral"]}
Organismal features:
{out_dict["specific"]}
Biodiversity Knowledge:
{out_dict["biodiversity"]}
Commentary:
{out_dict["commentary"]}
        """


def clean_string_output(out_dict: Dict[str, Union[str, Dict[str, str]]]) -> str:
    """
    Generate a formatted string of taxonomic classification and intermediate results.

    Args:
        out_dict (dict): A dict of output/intermediates from the RAG models with keys:
            - "guess_class" (dict): A dictionary of classification levels and values.
            - "caption" (str): A description or caption for the observation.
            - "ancestral" (str, optional): A description of ancestral features
              (if present).
            - "specific" (str, optional): A description of organismal features
              (if present).
            - "commentary" (str, optional): Additional commentary on the observation
              (if present).
            - "biodiversity" (str): Information about biodiversity knowledge.

    Returns
    -------
        str: A formatted string summarizing the observation, including classification,
        ancestral and organismal features (if available), commentary, and biodiversity
        knowledge.
    """
    if isinstance(out_dict["guess_class"], dict):
        cls = out_dict["guess_class"]
    else:
        cls = {
            "Kingdom": "N/A",
            "Phylum": "N/A",
            "Class": "N/A",
            "Order": "N/A",
            "Family": "N/A",
            "Genus": "N/A",
            "Species": "N/A",
        }
    cls_out = "\n".join(
        [f"{level}: {cls[level]}" for level in cls if cls[level] != "N/A"]
    )

    if out_dict.get("ancestral"):
        output = f"""
Caption:
{out_dict["caption"]}
{"=" * 50}
This observation is a:
{cls_out}
Ancestral features:
{out_dict["ancestral"]}
Organismal features:
{out_dict["specific"]}
Commentary:
{out_dict["commentary"]}
Biodiversity Knowledge:
{out_dict["biodiversity"]}
        """
    else:
        output = f"""
Caption:
{out_dict["caption"]}
{"=" * 50}
This observation is a:
{cls_out}
Biodiversity Knowledge:
{out_dict["biodiversity"]}
        """
    return output


def b64_to_pil(image_b64: str) -> Image.Image:
    """
    Convert a base64-encoded image to a PIL Image object.

    Args:
        image_b64 (bytes): The base64-encoded image data.

    Returns
    -------
        PIL.Image.Image: The decoded image as a PIL Image object.

    Raises
    ------
        SystemExit: If the image cannot be decoded or opened, the program exits
        with an error message.
    """
    try:
        im_file = BytesIO(image_b64.encode("utf-8"))  # Convert string to bytes
        image = Image.open(im_file)  # to PIL Image object
    except Exception as e:
        print(e)
        print("Unable to query with this image")
        sys.exit(1)
    return image


def imgurl_tob64(image_url: str) -> str:
    """
    Convert an image from a given URL to a Base64-encoded string.

    Args:
        image_url (str): The URL of the image to be converted.

    Returns
    -------
        str: The Base64-encoded string representation of the image.

    Raises
    ------
        Exception: If there is an error during the image retrieval or encoding
                   process, the exception is printed, and the program exits with
                   a status code of 1.
    """
    try:
        image_b64 = base64.b64encode(requests.get(image_url).content).decode("utf-8")
    except Exception as e:
        print(e)
        print("Unable to query with this image")
        sys.exit(1)
    return image_b64


def imgfile_tob64(image_path: str) -> str:
    """
    Convert an image file to a Base64-encoded string.

    Args:
        image_path (str): The file path to the image.

    Returns
    -------
        str: The Base64-encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pilimg_tob64(image_obj: Image.Image) -> str:
    """
    Convert a PIL Image object to a Base64-encoded string.

    Args:
        image_obj (PIL.Image.Image): The PIL Image object to be converted.

    Returns
    -------
        str: The Base64-encoded string representation of the image.
    """
    buffer = BytesIO()
    # Save the image to the buffer in a JPEG
    image_obj.save(buffer, format="JPEG")
    # Get the bytes data from the buffer
    image_bytes = buffer.getvalue()
    # Encode the bytes to base64
    return base64.b64encode(image_bytes).decode("utf-8")


def get_metrics(
    y_trues: List[str], y_preds: List[str], level: str, verbose: bool = True
) -> Dict[str, Union[float, int]]:
    """
    Calculate and return accuracy + F1 score metrics for true and predicted labels.

    Args:
        y_trues (list): List of true labels.
        y_preds (list): List of predicted labels.
        level (str): Taxonomic rank or level being evaluated.
        verbose (bool, optional): If True (default), prints detailed metrics to console.

    Returns
    -------
        dict: A dictionary containing:
            - "accuracy" (float): Accuracy score of the predictions.
            - "f1" (float): Weighted F1 score of the predictions.

    Notes
    -----
        - If both `y_trues` and `y_preds` are empty, returns NaN for both metrics.
        - Uses `LabelEncoder` to encode string labels into ints for metric computation.
    """
    le = LabelEncoder()
    le.fit(y_trues + y_preds)  # Fit label encoder to strings

    # Transform string labels to integers - can map back with le.inverse_transform()
    y_true_encoded = le.transform(y_trues)
    y_pred_encoded = le.transform(y_preds)

    if len(y_trues) == len(y_preds) == 0:  # If empty lists
        if verbose:
            print(f"Rank: {level}")
            print("=" * 40)
            print("No data available to calculate, using NaN")
        return {"accuracy": np.nan, "f1": np.nan}

    # Calculate metrics
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    f1 = f1_score(y_true_encoded, y_pred_encoded, average="weighted")

    if verbose:  # Report metrics with tax rank to screen
        count = len(y_preds)
        print(f"Rank: {level} ({count})")
        print("=" * 40)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("=" * 40)

    return {"accuracy": accuracy, "f1": f1}


def dict_match(
    y_true_dict: Dict[str, str], y_pred_dict: Dict[str, str]
) -> Tuple[int, int]:
    """
    Compare two dictionaries to determine the number of matching key-value pairs.

    Args:
        y_true_dict (dict): The ground truth dictionary containing the correct
            key-value pairs.
        y_pred_dict (dict): The predicted dictionary to compare against the
            ground truth.

    Returns
    -------
        tuple: A tuple (correct, total) where:
            - correct (int): The number of key-value pairs that match between
              the two dictionaries.
            - total (int): The total number of key-value pairs in the predicted
              dictionary.
    """
    # Determine number of ranks predicted and number of ranks correct
    total = len(y_pred_dict)
    correct = sum([y_true_dict[key] == y_pred_dict[key] for key in y_pred_dict])
    return correct, total


def classify_report(
    true_dicts: List[Dict[str, str]],
    pred_dicts: List[Dict[str, str]],
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Generate classification metrics for taxonomic predictions at various ranks.

    This function compares true taxonomic classifications with predicted classifications
    and calculates metrics such as precision, recall, and F1-score for each tax rank.

    Args:
        true_dicts: list[dict[str,str]] containing the true tax classifications.
                Each dict with taxonomic ranks as keys (e.g., "Kingdom", "Phylum").
        pred_dicts: list[dict[str,str]] containing the predicted tax classifications.
                Each dict with taxonomic ranks as keys (e.g., "Kingdom", "Phylum").
        verbose (bool, optional): If True, additional info printed during calculation.
                Defaults to True.

    Returns
    -------
        dict[str, dict[str,float]]: Rank-wise classification metrics.
              Each taxonomic rank is sub-dictionary with:
              - "Count": The number of predictions made for that rank.
              - Additional keys for metrics such as precision, recall, and F1-score.

    Notes
    -----
        - Predictions for ranks not included in the `ranks` list will be ignored.
        - The `dict_match` and `get_metrics` helper functions used to calculate
          the num correct predictions and the classification metrics, respectively.
    """
    ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    out_dict = {}
    for rank in ranks:  # Go through each rank and build classification metrics dict
        true_names = [
            true_dict[rank]
            for true_dict, pred_dict in zip(true_dicts, pred_dicts)
            if rank in pred_dict
        ]
        pred_names = [pred_dict[rank] for pred_dict in pred_dicts if rank in pred_dict]
        out_dict[rank] = {"Count": float(len(pred_names))}
        metrics = get_metrics(true_names, pred_names, rank, verbose=verbose)
        # Convert integer values in metrics to floats
        metrics = {k: float(v) if isinstance(v, int) else v for k, v in metrics.items()}
        out_dict[rank].update(metrics)
    return out_dict


def custom_collate_fn(
    batch: List[Tuple[Any, Dict[str, Any]]],
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Process batches of data using a custom collate function.

    This function is typically used in data loaders to handle batches of data
    where each item in the batch is a tuple containing an image and a class dictionary.
    It separates the images and class dictionaries into two separate lists.

    Args:
        batch (list of tuple): A batch of data where each element is a tuple
            (image, class_dict). `image` is the image data, and `class_dict`
            is a dictionary containing class-related information.

    Returns
    -------
        tuple: A tuple containing two lists:
            - A list of images.
            - A list of class dictionaries.
    """
    images, class_dicts = zip(*batch)
    return list(images), list(class_dicts)


async def rag_evaluate(
    eval_dict: Dict[str, Any], embeddings: Any
) -> Optional[pd.DataFrame]:
    """
    Evaluate the quality of a response in a Retrieval-Augmented Generation (RAG) system.

    This function takes an output dict and embeddings, formats it and computes
    RAGAS scores for faithfulness and response relevancy using an LLM and embeddings.

    Args:
        eval_dict (dict): A dictionary containing the evaluation data. It must include:
            - "caption" (str): The caption describing the new organism.
            - "context" (str): The context that may or may not match the caption.
        embeddings: Pre-trained embedding model used for RAG.

    Returns
    -------
        pandas.DataFrame: A DataFrame of faithfulness and response relevancy scores.

    Notes
    -----
        - The function uses an LLM evaluator of response's faithfulness and relevancy.
        - The evaluation pipeline formats the input, creates a sample, and scores it.
        - The returned DataFrame contains the computed scores for each metric.
    """
    if eval_dict is None:
        return None
    # Formatting all elements together
    eval_dict["response"] = clean_string_output(eval_dict)

    # Build dictionary needed for ragas evaluation
    eval_data = {
        "user_input": f"""
            You are an expert AI taxonomist. Your task is to use the organisms discussed in the caption of a new organism to generate a taxonomic classification for the new organism.

            You will also be provided with some context that could or could not match the caption, if there is information in the context that matches the caption, you can use that info to inform your decision about the taxonomic classification, otherwise, if information does not match the details provided in the caption, disregard it.

            You will be provided with context and a caption, provide in your response:
            1. A Taxonomic classification
            2. A description pairing physical traits common to both the new organism described in the caption and other organisms described in the context that indicate and support the choice made in the taxonomic classification.
            3. A description of physical traits particular to this new organism described in the caption. These traits may set it apart from other organisms, may suggest it has unique features, and/or contain traits that may be candidates to investigate for a more specific taxonomic classification.
            4. Commentary on your choice, including discussion of confidence, what new information about the new organism would help support the taxonomic classification and what new information would dispute the taxonomic classification.
            5. A few paragraphs describing the features present (from the caption) and how they could relate to the biodiversity knowledge that is relevant to the taxa chosen.

            Do not include a taxonomic classification for a certain rank unless you are confident from the caption (and/or context) about the classification.

            <caption>
            {eval_dict["caption"]}
            </caption>
        """,
        "response": eval_dict["response"],
        "retrieved_contexts": eval_dict["context"],
    }
    sample = SingleTurnSample(
        user_input=eval_data["user_input"],
        response=eval_data["response"],
        retrieved_contexts=eval_data["retrieved_contexts"],
    )
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
    wrapped_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    faith_scorer = Faithfulness(llm=wrapped_llm)
    relevance_scorer = ResponseRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings)
    print("Fetching RAGAS scores..")
    scores = {
        "faithfulness": [await faith_scorer.single_turn_ascore(sample)],
        "response_relevancy": [await relevance_scorer.single_turn_ascore(sample)],
    }
    return pd.DataFrame.from_dict(scores)


def write_overall_metrics(csv_filename: str, data: Dict[str, Dict[str, float]]) -> None:
    """
    Write overall metrics to a CSV file.

    This function takes a dictionary of metrics and writes them to a CSV file.
    The CSV file will include a header row with "Rank", "Accuracy", and "F1".
    For each rank in the data, it writes the corresponding accuracy and F1 score
    if available, or just the rank and a single metric if the data is not a dictionary.

    Args:
        csv_filename (str): The path to the CSV file where the metrics will be written.
        data (dict): A dictionary containing metrics for each rank. Each key is a rank,
                     and the value is either:
                     - A dictionary with "accuracy" and "f1" keys.
                     - A single value representing a metric.

    Raises
    ------
        IOError: If there is an issue writing to the file.
    """
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Rank", "Accuracy", "F1"])

        # Write each rank's metrics
        for rank, metrics in data.items():
            if isinstance(metrics, dict):  # For ranks with 'accuracy' and 'f1'
                writer.writerow(
                    [rank, metrics.get("accuracy", ""), metrics.get("f1", "")]
                )
            else:  # For PropRanksCorrect and Ranks
                writer.writerow([rank, metrics, ""])


# Write rank-level attempts (Count) to CSV
def write_rank_attempts_csv(
    csv_filename: str, data: Dict[str, Dict[str, float]], total_samples: int
) -> None:
    """Write per-rank attempts with percentage to CSV.

    Header: ["Rank", "Attempts (%)"]. Each cell shows "{count} ({percent})",
    where percent is count/total_samples formatted as a percentage with no decimals,
    e.g., "24 (100%)".
    """
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Attempts (%)"])
        denom = max(int(total_samples), 0)
        for rank, metrics in data.items():
            count_val = metrics.get("Count", 0) if isinstance(metrics, dict) else 0
            try:
                count_int = int(count_val)
            except Exception:
                count_int = 0
            if denom > 0:
                ratio = count_int / denom
                percent_str = format(ratio, ".0%")
            else:
                percent_str = "0%"
            cell = f"{count_int} ({percent_str})"
            writer.writerow([rank, cell])


# Writer for per-sample BinaryAccuracy labels
def write_sample_binary_accuracy_csv(
    guess_classes: List[Dict[str, Any]], csv_filename: str
) -> None:
    """Write per-sample binary accuracy labels to CSV.

    If RSID is present in the guess dictionaries, write columns [RSID, BinaryAccuracy].
    Otherwise, write [Index, BinaryAccuracy] where Index is the sample index in the list.
    """
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        has_rsid = any("RSID" in g for g in guess_classes)
        if has_rsid:
            writer.writerow(["RSID", "BinaryAccuracy"])
            for g in guess_classes:
                val = g.get("BinaryAccuracy", "")
                val = "" if val is None else val
                writer.writerow([g.get("RSID", ""), val])
        else:
            writer.writerow(["Index", "BinaryAccuracy"])
            for i, g in enumerate(guess_classes):
                val = g.get("BinaryAccuracy", "")
                val = "" if val is None else val
                writer.writerow([i, val])


# Per-sample binary accuracy across all ranks, this function is applied to each sample
def sample_binary_accuracy(true_class: Dict[str, str], pred_class: Dict[str, str]) -> Optional[int]:
    """Compute binary accuracy for a single sample on predicted-only ranks.

    Returns 1 if and only if every predicted rank that exists in the true_class
    matches exactly. Returns 0 if any predicted rank mismatches. If no ranks are
    predicted (i.e., model abstains completely), return None.
    """
    canonical_ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    predicted_ranks = [r for r in canonical_ranks if r in pred_class]
    if len(predicted_ranks) == 0:
        return None
    for rank in predicted_ranks:
        # If true lacks the rank, we treat it as not comparable; skip
        if rank not in true_class:
            continue
        if pred_class[rank] != true_class[rank]:
            return 0
    return 1


def extract_tax_metrics(
    result_obj: List[Dict[str, Any]], verbose: bool = True
) -> Tuple[Dict[str, Dict[str, Union[float, int]]], List[Dict[str, str]]]:
    """
    Extract classification metrics from a result object.

    This function processes a list of dictionaries containing true and guessed
    class labels, computes a classification report, and returns the report
    along with the guessed class labels.

    Args:
        result_obj (list of dict): A list of dictionaries where each dictionary
            contains the keys "true_class" and "guess_class" representing the
            true and predicted class labels, respectively.

    Returns
    -------
        tuple: A tuple containing:
            - class_report (dict): A classification report generated by
              `classify_report` summarizing the performance metrics.
            - guess_classes (list): A list of guessed class labels extracted
              from the input.
    """
    true_classes = [out_dict["true_class"] for out_dict in result_obj]
    guess_classes = [out_dict["guess_class"] for out_dict in result_obj]
    class_report = classify_report(true_classes, guess_classes, verbose=verbose)
    # Attach per-sample BinaryAccuracy
    for true_class, guess_class in zip(true_classes, guess_classes):
        try:
            guess_class["BinaryAccuracy"] = sample_binary_accuracy(true_class, guess_class)
        except Exception:
            guess_class["BinaryAccuracy"] = 0
    return class_report, guess_classes


def extract_tax_metrics_rs(
    result_obj: List[Dict[str, Any]], verbose: bool = True
) -> Tuple[Dict[str, Dict[str, Union[float, int]]], List[Dict[str, str]]]:
    """
    Extract taxonomic metrics and enrich guess class dictionaries with RSID information.
    """
    true_classes = [out_dict["true_class"] for out_dict in result_obj]
    guess_classes = [out_dict["guess_class"] for out_dict in result_obj]
    class_report = classify_report(true_classes, guess_classes, verbose=verbose)
    rsids = [out_dict["RSID"] for out_dict in result_obj]
    # Add RSID and BinaryAccuracy to each guess_class dictionary
    for true_class, guess_class, rsid in zip(true_classes, guess_classes, rsids):
        guess_class["RSID"] = rsid
        try:
            guess_class["BinaryAccuracy"] = sample_binary_accuracy(true_class, guess_class)
        except Exception:
            guess_class["BinaryAccuracy"] = 0
    return class_report, guess_classes


def filter_nonempty_results(result_obj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out samples whose predicted taxonomy is empty across all ranks.

    A sample is kept if its "guess_class" contains at least one of the canonical
    taxonomy ranks: Kingdom, Phylum, Class, Order, Family, Genus, Species.
    """
    canonical_ranks = {"Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"}
    filtered: List[Dict[str, Any]] = []
    for sample in result_obj:
        guess = sample.get("guess_class", {}) or {}
        has_any_rank = any(rank in guess for rank in canonical_ranks)
        if has_any_rank:
            filtered.append(sample)
    return filtered


def validate_uq_outputs(
    predictions_csv: str,
    sample_binary_csv: str,
    prompt_jsonl_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate consistency of filtered UQ outputs.

    Checks:
    - Row counts of predictions vs. sample-binary CSVs match
    - If both have "RSID" columns, their RSID sets match exactly
    - If prompt_jsonl_path is provided, ensure RSIDs in predictions are a subset
      of RSIDs in JSONL log
    Returns a dict with keys: {"ok": bool, "messages": List[str]}.
    """
    messages: List[str] = []
    ok = True

    try:
        preds_df = pd.read_csv(predictions_csv)
    except Exception as e:
        return {"ok": False, "messages": [f"Failed to read predictions CSV: {e}"]}
    try:
        bin_df = pd.read_csv(sample_binary_csv)
    except Exception as e:
        return {"ok": False, "messages": [f"Failed to read sample-binary CSV: {e}"]}

    if len(preds_df) != len(bin_df):
        ok = False
        messages.append(
            f"Row count mismatch: predictions={len(preds_df)} vs sample_binary={len(bin_df)}"
        )

    if "RSID" in preds_df.columns and "RSID" in bin_df.columns:
        preds_rsids = set(preds_df["RSID"].astype(str))
        bin_rsids = set(bin_df["RSID"].astype(str))
        if preds_rsids != bin_rsids:
            ok = False
            only_in_preds = sorted(preds_rsids - bin_rsids)
            only_in_bin = sorted(bin_rsids - preds_rsids)
            if only_in_preds:
                messages.append(f"RSIDs only in predictions: {only_in_preds[:20]}" + (" ..." if len(only_in_preds) > 20 else ""))
            if only_in_bin:
                messages.append(f"RSIDs only in sample_binary: {only_in_bin[:20]}" + (" ..." if len(only_in_bin) > 20 else ""))

    if prompt_jsonl_path:
        try:
            jsonl_rsids: set[str] = set()
            with open(prompt_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        rsid_val = obj.get("RSID")
                        if rsid_val is not None and str(rsid_val) != "":
                            jsonl_rsids.add(str(rsid_val))
                    except Exception:
                        continue
            if "RSID" in preds_df.columns:
                preds_rsids = set(preds_df["RSID"].astype(str))
                if not preds_rsids.issubset(jsonl_rsids):
                    ok = False
                    missing = sorted(preds_rsids - jsonl_rsids)
                    messages.append(
                        f"Some prediction RSIDs not found in JSONL log: {missing[:20]}" + (" ..." if len(missing) > 20 else "")
                    )
        except Exception as e:
            ok = False
            messages.append(f"Failed to read/parse JSONL log: {e}")

    return {"ok": ok, "messages": messages}


def write_preds_to_csv(guess_classes: List[Dict[str, str]], csv_filename: str) -> None:
    """
    Write prediction data to a CSV file.

    If a "BinaryAccuracy" field is present in the guess dictionaries, an
    additional column will be written after "Species".
    """
    # Write guess_classes along with image paths to a new CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Determine if BinaryAccuracy column is present
        include_binary = any("BinaryAccuracy" in guess for guess in guess_classes)
        # Write header for guess classes if the file is new
        if file.tell() == 0:
            header = [
                "RSID",
                "Kingdom",
                "Phylum",
                "Class",
                "Order",
                "Family",
                "Genus",
                "Species",
            ]
            if include_binary:
                header.append("BinaryAccuracy")
            writer.writerow(header)

        for guess in guess_classes:
            row = [
                guess.get("RSID", ""),
                guess.get("Kingdom", ""),
                guess.get("Phylum", ""),
                guess.get("Class", ""),
                guess.get("Order", ""),
                guess.get("Family", ""),
                guess.get("Genus", ""),
                guess.get("Species", ""),
            ]
            if include_binary:
                val = guess.get("BinaryAccuracy", "")
                val = "" if val is None else val
                row.append(val)
            writer.writerow(row)


def hierarchical_metrics(
    true_dicts: List[Dict[str, str]],
    pred_dicts: List[Dict[str, str]],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Calculate hierarchical metrics for taxonomic predictions.

    Compares true taxonomic classifications with predicted classifications and
    calculates the prediction set's hierarchical precision (hP), hierarchical
    recall (hR), and hierarchical F1 (hF) [(Snæbjarnarson et al., 2025)](https://arxiv.org/abs/2504.05457).

    Keys in each dict should be the standard Linnaean ranks from Phylum and below,
    (e.g. "Phylum", "Class", ..., "Species"), but is case-insensitive.

    Args:
        true_dicts: list[dict[str,str]] mapping rank -> true taxon name.
        pred_dicts: list[dict[str,str]] mapping rank -> predicted taxon name.
        verbose (bool, optional): If True, prints these metrics to stdout.

    Returns
    -------
        dict[str, float] with keys:
            - "hp": hierarchical precision
            - "hr": hierarchical recall
            - "hf": hierarchical F1
    """
    # Normalize
    ranks = [rank.lower() for rank in RANKS]
    true_dicts = [
        {rank.lower(): val for rank, val in true_dict.items()}
        for true_dict in true_dicts
    ]
    pred_dicts = [
        {rank.lower(): val for rank, val in pred_dict.items()}
        for pred_dict in pred_dicts
    ]

    n = len(true_dicts)
    hp_sum = 0.0
    hr_sum = 0.0

    print("Before for loop")
    print("GT:", true_dicts)
    print("PRED:", pred_dicts)

    count = 0
    for true_dict, pred_dict in zip(true_dicts, pred_dicts):
        count += 1
        print("count:", count)
        # Build the ancestor paths for true and predicted in a single loop
        anc_true = []
        anc_pred = []

        for rank in ranks:
            true_taxa = true_dict.get(rank, "")
            pred_taxa = pred_dict.get(rank, "")
            # Check and append to anc_true
            if true_taxa.strip():
                anc_true.append(true_taxa.lower())
                print(rank, "GT added")
            else:
                # Stop adding to anc_true if a rank is missing
                print(rank, "GT skipped")
                break
            # Check and append to anc_pred
            if pred_taxa.strip():
                anc_pred.append(pred_taxa.lower())
                print(rank, "pred added")
            else:
                # Stop adding to anc_pred if a rank is missing
                print(rank, "pred skipped")
                break

        print("Afer for loop")
        set_true = set(anc_true)
        print("GT:", set_true)
        set_pred = set(anc_pred)
        print("PRED:", set_pred)
        inter_size = len(set_true & set_pred)

        # Precision: |intersection| / |predicted path|
        hp_i = inter_size / len(set_pred) if set_pred else 0.000
        # Recall: |intersection| / |true path|
        hr_i = inter_size / len(set_true) if set_true else 0.000

        hp_sum += hp_i
        hr_sum += hr_i

    # Sum over all examples
    hp = hp_sum / n
    hr = hr_sum / n
    hf = (2 * hp * hr / (hp + hr)) if (hp + hr) > 0 else 0.0

    if verbose:
        print(f"Hierarchical Precision (hp): {hp:.3f}")
        print(f"Hierarchical Recall (hr): {hr:.3f}")
        print(f"Hierarchical F1 (hf): {hf:.3f}")

    return {"hp": hp, "hr": hr, "hf": hf}


def sample_hierarchical_metrics(true_dict: Dict[str, str], pred_dict: Dict[str, str]) -> Dict[str, float]:
    """
    Compute per-sample hierarchical precision/recall/F1 (HP/HR/HF) using the same
    logic as `hierarchical_metrics` (current implementation based on set intersection
    of built ancestor paths with early stopping on first missing rank).
    """
    ranks = [rank.lower() for rank in RANKS]
    true_d = {k.lower(): v for k, v in (true_dict or {}).items()}
    pred_d = {k.lower(): v for k, v in (pred_dict or {}).items()}

    anc_true: List[str] = []
    anc_pred: List[str] = []
    for rank in ranks:
        t_val = true_d.get(rank, "")
        p_val = pred_d.get(rank, "")
        if t_val.strip():
            anc_true.append(t_val.lower())
        else:
            break
        if p_val.strip():
            anc_pred.append(p_val.lower())
        else:
            break

    set_true = set(anc_true)
    set_pred = set(anc_pred)
    inter_size = len(set_true & set_pred)

    hp_i = inter_size / len(set_pred) if len(set_pred) > 0 else 0.0
    hr_i = inter_size / len(set_true) if len(set_true) > 0 else 0.0
    hf_i = (2 * hp_i * hr_i / (hp_i + hr_i)) if (hp_i + hr_i) > 0 else 0.0
    return {"HP": hp_i, "HR": hr_i, "HF": hf_i}


def write_sample_hierarchical_metrics_csv(
    samples: List[Dict[str, Any]], csv_filename: str
) -> None:
    """
    Write per-sample hierarchical metrics (HP, HR, HF) to CSV.
    If RSID present in any sample, first column is RSID; otherwise Index.
    """
    has_rsid = any(s.get("RSID") is not None for s in samples)
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        if has_rsid:
            writer.writerow(["RSID", "HP", "HR", "HF"])
            for s in samples:
                rsid = s.get("RSID", "")
                metrics = sample_hierarchical_metrics(s.get("true_class", {}), s.get("guess_class", {}))
                writer.writerow([rsid, metrics["HP"], metrics["HR"], metrics["HF"]])
        else:
            writer.writerow(["Index", "HP", "HR", "HF"])
            for i, s in enumerate(samples):
                metrics = sample_hierarchical_metrics(s.get("true_class", {}), s.get("guess_class", {}))
                writer.writerow([i, metrics["HP"], metrics["HR"], metrics["HF"]])


def write_preds_hierarchical_to_csv(
    samples: List[Dict[str, Any]], csv_filename: str
) -> None:
    """
    Write predictions with appended per-sample hierarchical metrics (HP, HR, HF).
    If RSID present in any sample, first column is RSID; otherwise Index.
    """
    has_rsid = any(s.get("RSID") is not None for s in samples)
    cols = RANKS.copy()
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = (["RSID"] if has_rsid else ["Index"]) + cols + ["HP", "HR", "HF"]
        writer.writerow(header)
        for i, s in enumerate(samples):
            guess = s.get("guess_class", {}) or {}
            row_id = s.get("RSID", "") if has_rsid else i
            metrics = sample_hierarchical_metrics(s.get("true_class", {}), guess)
            row = [row_id] + [guess.get(col, "") for col in cols] + [metrics["HP"], metrics["HR"], metrics["HF"]]
            writer.writerow(row)
