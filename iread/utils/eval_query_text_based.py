import torch
import torch.nn.functional as F
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from utils.eval_single_exp import metric_parser, get_experiment_name
from utils.eval_per_query import calculate_evaluation
from utils.plots import plot_query_performance_vs_query_length
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)


download_nltk_resources()


@st.cache_resource
def tokenize_text(text):
    return tokenizer.tokenize(text)


@st.cache_data
def get_query_len(selected_queries):
    # Tokenize each query_text and calculate the length of tokens
    selected_queries['tokens'] = selected_queries['query_text'].apply(tokenize_text)
    selected_queries['token_len'] = selected_queries['tokens'].apply(len)

    # Create the dictionary with query_id as key and len of tokens as value
    token_len_dict = {row['query_id']: row['token_len'] for index, row in selected_queries.iterrows()}
    return token_len_dict


@st.cache_data
def add_token_lengths(results, token_lengths):
    for experiment in results:
        # Find the first measure that's not 'token_length'
        first_measure = next(measure for measure in results[experiment] if measure != 'token_length')

        # Use this measure to determine the number of queries
        num_queries = len(results[experiment][first_measure][first_measure])

        results[experiment]['token_length'] = {'token_length': []}
        for i in range(num_queries):
            query_id = str(i + 1)  # Assuming query IDs start from 1
            if query_id in token_lengths:
                results[experiment]['token_length']['token_length'].append(token_lengths[query_id])
            else:
                # If token length is not available, use a placeholder value (e.g., -1)
                results[experiment]['token_length']['token_length'].append(-1)
    return results


def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


@st.cache_data
def per_query_length_evaluation(qrel, runs, selected_queries, metric_list, relevance_threshold, selected_cutoff, baseline_run, threshold_value):
    results_per_run = {}
    parsed_metrics = []

    # Parse each metric in metric_list
    for metric in metric_list:
        parsed_metric = metric_parser(metric, relevance_threshold, selected_cutoff)
        parsed_metrics.append(parsed_metric)

    if baseline_run is None and threshold_value is None:
        # Calculate evaluation results for each other experiment and collect p-values
        for run_name, run_data in runs.items():
            experiment = get_experiment_name(run_name, baseline_run)
            results_per_run[experiment] = {}
            for parsed_metric in parsed_metrics:
                results_per_run[experiment][str(parsed_metric)] = calculate_evaluation(parsed_metric, qrel, run_data)

        query_length = get_query_len(selected_queries)
        results = add_token_lengths(results_per_run, query_length)
        # Call the plot function per measure scores
        plot_query_performance_vs_query_length(results)

        return results


@st.cache_resource()
def get_huggingface_model(model_name):
    # Load model & tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
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
    pca = TSNE(n_components=2)
    pca_result = pca.fit_transform(embeddings_array)
    pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    pca_df["query_id"] = query_data["query_id"]
    pca_df["query_type"] = query_data["query_type"]
    return pca_df
