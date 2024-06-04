import pandas as pd
import streamlit as st
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


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
