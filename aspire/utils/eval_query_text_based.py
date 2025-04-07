import re
from typing import Dict, List

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from utils.eval_multiple_exp import calculate_evaluation
from utils.eval_single_exp import get_experiment_name, metric_parser

# Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


@st.cache_resource
def tokenize_text(text):
    return text.split()
    # return tokenizer.tokenize(text)


@st.cache_data
def get_query_len(selected_queries):
    # Tokenize each query_text and calculate the length of tokens
    selected_queries["tokens"] = selected_queries["query_text"].apply(tokenize_text)
    selected_queries["token_len"] = selected_queries["tokens"].apply(len)

    # Create the dictionary with query_id as key and len of tokens as value
    token_len_dict = {
        str(row["query_id"]): row["token_len"] for index, row in selected_queries.iterrows()
    }
    return token_len_dict


@st.cache_data
def add_token_lengths(results, token_lengths):
    for experiment in results:
        # Find the first measure that's not 'token_length'
        first_measure = next(
            measure for measure in results[experiment] if measure != "token_length"
        )

        # Initialize token_length dictionary for this experiment
        results[experiment]["token_length"] = {"token_length": {}}

        # Iterate through each query in the first measure
        for query_id in results[experiment][first_measure][first_measure]:
            # Convert query_id to string if it's not already
            query_id_str = str(query_id)

            if query_id_str in token_lengths:
                results[experiment]["token_length"]["token_length"][query_id_str] = token_lengths[query_id_str]
            else:
                # If token length is not available, use a placeholder value (e.g., -1)
                results[experiment]["token_length"]["token_length"][query_id_str] = -1

    return results


@st.cache_data
def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r"[^\w\s]", "", text.lower())
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


@st.cache_data
def per_query_length_evaluation(
        qrel,
        runs,
        selected_queries,
        metric_list,
        relevance_threshold,
        selected_cutoff,
        baseline_run,
        threshold_value,
):
    results_per_run = {}
    parsed_metrics = []

    # Parse each metric in metric_list
    for metric in metric_list:
        parsed_metric = metric_parser(metric, relevance_threshold, selected_cutoff)
        parsed_metrics.append(parsed_metric)

    # Get query lengths
    query_length = get_query_len(selected_queries)

    # Calculate evaluation results for each experiment
    for run_name, run_data in runs.items():
        experiment = get_experiment_name(run_name, baseline_run)
        results_per_run[experiment] = {}
        for parsed_metric in parsed_metrics:
            results = calculate_evaluation(parsed_metric, qrel, run_data)
            for metric_name, metric_data in results.items():
                results_per_run[experiment][metric_name] = {
                    metric_name: metric_data['values'],
                    'query_id': metric_data['query_ids']
                }

        # Add token lengths to results
        results_per_run[experiment]['token_length'] = {
            'token_length': {str(qid): query_length.get(str(qid), -1) for qid in metric_data['query_ids']}
        }

    return results_per_run


@st.cache_data
def query_clf_relevance_assessments(
    data: Dict[str, Dict],
    method: str = "Median Absolute Deviation",
    threshold: float = 3.5,
) -> Dict[str, Dict[str, List[str]]]:
    query_ids = list(data.keys())
    metrics = ["irrelevant"] + list(next(iter(data.values()))["relevant"].keys())

    # Pre-allocate numpy arrays
    values = np.zeros((len(query_ids), len(metrics)))

    # Populate the array in a single pass through the data
    for i, (query_id, query_data) in enumerate(data.items()):
        values[i, 0] = query_data["irrelevant"]
        for j, metric in enumerate(metrics[1:], start=1):
            values[i, j] = query_data["relevant"][metric]

    results = {}

    for j, metric in enumerate(metrics):
        metric_values = values[:, j]

        if method == "Median Absolute Deviation":
            median = np.median(metric_values)
            mad = np.median(np.abs(metric_values - median))
            lower_bound = median - threshold * mad
            upper_bound = median + threshold * mad
        elif method == "Percentile-based method":
            lower_bound, upper_bound = np.percentile(metric_values, [5, 95])
        elif method == "Modified Z-score":
            median = np.median(metric_values)
            mad = np.median(np.abs(metric_values - median))
            # modified_z_scores = 0.6745 * (metric_values - median) / mad
            lower_bound = median - threshold * mad
            upper_bound = median + threshold * mad
        elif method == "Max-Min-Median":
            sorted_indices = np.argsort(metric_values)
            high = sorted_indices[-5:]
            low = sorted_indices[:5]
            median = np.median(metric_values)
            middle = sorted_indices[np.abs(metric_values - median).argsort()[:5]]
            results[metric] = {
                "5_queries_most_assessments": [query_ids[i] for i in high],
                "5_queries_around_median_assessments": [query_ids[i] for i in middle],
                "5_queries_least_assessments": [query_ids[i] for i in low],
            }
            continue
        else:
            raise ValueError("Invalid method specified")

        if method != "Max-Min-Median":
            high = np.where(metric_values > upper_bound)[0]
            low = np.where(metric_values < lower_bound)[0]
            normal = np.where(
                (metric_values >= lower_bound) & (metric_values <= upper_bound)
            )[0]

            if method == "Percentile-based method":
                results[metric] = {
                    "queries_above_95th_percentile": [query_ids[i] for i in high],
                    "queries_between_5th_95th_percentiles": [
                        query_ids[i] for i in normal
                    ],
                    "queries_below_5th_percentile": [query_ids[i] for i in low],
                }
            else:
                results[metric] = {
                    "queries_above_threshold": [query_ids[i] for i in high],
                    "queries_within_normal_range": [query_ids[i] for i in normal],
                    "queries_below_threshold": [query_ids[i] for i in low],
                }

    return results


@st.cache_data
def remove_stopwords_from_queries(
    queries_df, text_column="query_text", id_column="query_id"
):
    """
    Remove stopwords, numbers, and punctuation from queries in a DataFrame using a faster method.

    Parameters:
    queries_df (pd.DataFrame): DataFrame containing queries
    text_column (str): Name of the column containing query text
    id_column (str): Name of the column containing query IDs

    Returns:
    pd.DataFrame: DataFrame with stopwords, numbers, and punctuation removed from the specified text column
    """
    # Ensure the required columns exist
    if text_column not in queries_df.columns or id_column not in queries_df.columns:
        raise ValueError(
            f"Required columns {text_column} and/or {id_column} not found in DataFrame"
        )

    # Get English stopwords
    stop_words = set(stopwords.words("english"))

    # Compile regex pattern for word boundary, excluding numbers and punctuation
    word_pattern = re.compile(r"\b[a-zA-Z]+\b")

    # Create a vectorized function to remove stopwords, numbers, and punctuation
    def clean_text_vectorized(text):
        return " ".join(
            [
                word
                for word in word_pattern.findall(text.lower())
                if word not in stop_words
            ]
        )

    # Create a new DataFrame to avoid modifying the original
    result_df = queries_df.copy()

    # Apply cleaning to the text column using vectorized operation
    result_df[text_column] = result_df[text_column].apply(clean_text_vectorized)

    return result_df


@st.cache_resource
def downloading_huggingface_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer_emb = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer_emb


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


@st.cache_resource()
def get_embeddings(query_data, _model, _tokenizer):
    embeddings = []

    for query in query_data["query_text"]:
        # Tokenize the query
        encoded_input = _tokenizer(
            query, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = _model(**encoded_input)

        # Perform pooling and normalization
        pooled_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)

        embeddings.append(normalized_embeddings.cpu().numpy())

    query_data["embeddings"] = [emb.flatten() for emb in embeddings]
    return query_data


@st.cache_data()
def perform_pca(query_data):
    # Aggregate embeddings if they are 2D (e.g., sequence of vectors)
    aggregated_embeddings = []
    for emb in query_data["embeddings"]:
        # Averaging across the sequence dimension
        if emb.ndim == 2:
            avg_emb = emb.mean(axis=0)
            aggregated_embeddings.append(avg_emb)
        else:
            aggregated_embeddings.append(emb)

    # Convert the list of embeddings to a 2D NumPy array
    embeddings_array = np.vstack(aggregated_embeddings)

    # Perform PCA
    pca = TSNE(n_components=3)
    pca_result = pca.fit_transform(embeddings_array)
    pca_df = pd.DataFrame(
        pca_result, columns=["Prin. Comp. 1", "Prin. Comp. 2", "Prin. Comp. 3"]
    )
    pca_df["query_id"] = query_data["query_id"]
    return pca_df


@st.cache_data
def query_similarity_performance(
    queries,
    qrel,
    runs,
    metric_list,
    selected_cutoff,
    relevance_threshold,
    embedding_model_name,
):
    parsed_metrics = []
    results_per_run = {}

    embedding_model, model_tokenizer = downloading_huggingface_model(
        embedding_model_name
    )

    queries = get_embeddings(queries, embedding_model, model_tokenizer)
    pca_df = perform_pca(queries)

    # Parse each metric in metric_list
    for metric in metric_list:
        parsed_metric = metric_parser(metric, relevance_threshold, selected_cutoff)
        parsed_metrics.append(parsed_metric)

    # Calculate evaluation results for each other experiment and collect p-values
    for run_name, run_data in runs.items():
        experiment = get_experiment_name(run_name, None)
        results_per_run[experiment] = {}
        for parsed_metric in parsed_metrics:
            results = calculate_evaluation(parsed_metric, qrel, run_data)
            for metric_name, metric_data in results.items():
                results_per_run[experiment][metric_name] = {
                    metric_name: [float(value) for value in metric_data['values']],
                    'query_id': metric_data['query_ids']
                }

    return pca_df, results_per_run
