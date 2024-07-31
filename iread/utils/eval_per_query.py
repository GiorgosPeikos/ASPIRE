import numpy as np
import streamlit
import streamlit as st
import statistics
import ir_measures
from utils.eval_single_exp import *


@st.cache_data
def get_frequently_retrieved_docs(runs, selected_cutoff):
    # Get unique query IDs and document IDs
    all_queries = set()
    all_docs = set()
    for run_data in runs.values():
        all_queries.update(run_data['query_id'])
        all_docs.update(run_data['doc_id'])

    query_map = {q: i for i, q in enumerate(all_queries)}
    doc_map = {d: i for i, d in enumerate(all_docs)}

    n_queries = len(all_queries)
    n_docs = len(all_docs)
    n_runs = len(runs)

    # Create a 3D numpy array to represent the data
    data = np.zeros((n_runs, n_queries, n_docs), dtype=bool)

    # Fill the array
    for i, (run_name, run_data) in enumerate(runs.items()):
        for _, row in run_data.iterrows():
            if row['rank'] <= selected_cutoff:
                q_idx = query_map[row['query_id']]
                d_idx = doc_map[row['doc_id']]
                data[i, q_idx, d_idx] = True

    # Find documents retrieved by all systems
    docs_retrieved_by_all = data.all(axis=0)

    # Get the actual document IDs and query IDs
    frequent_docs = []
    for q_idx, d_idx in zip(*np.where(docs_retrieved_by_all)):
        query_id = list(query_map.keys())[list(query_map.values()).index(q_idx)]
        doc_id = list(doc_map.keys())[list(doc_map.values()).index(d_idx)]
        frequent_docs.append((query_id, doc_id))

    return frequent_docs


@st.cache_data()
def per_query_evaluation(qrel, run, metric, cutoff, relevance_threshold):
    # Check if metric contains '@' and split if it does
    # Check if the measure supports a relevance threshold
    if metric in measures_with_rel_param and metric in measures_with_cutoff:
        # Format the new measure string with the relevance threshold
        new_metric_str = f"{metric}(rel={relevance_threshold})@{cutoff}"
    else:
        # For metrics without cutoff, use the base metric directly
        if metric in measures_with_rel_param:
            new_metric_str = f"{metric}(rel={relevance_threshold})"
        elif metric in measures_with_cutoff:
            new_metric_str = f"{metric}@{cutoff}"
        else:
            new_metric_str = metric

    # Parse the new metric using ir_measures
    parsed_metric = ir_measures.parse_measure(new_metric_str)

    # Calculate the evaluation using the parsed metric
    res_eval = list(ir_measures.iter_calc([parsed_metric], qrel, run))

    return parsed_metric, res_eval


@st.cache_data()
def good_bad_queries(res_eval):
    # Assuming res_eval is a list of dictionaries like: [{'Query ID': '1', 'Score': 0.1049}, ...]
    data = [
        {"Query ID": int(metric.query_id), "Score": metric.value} for metric in res_eval
    ]

    scores = [entry["Score"] for entry in data]
    ids = [entry["Query ID"] for entry in data]

    median = statistics.median(scores)

    average = statistics.mean(scores)

    return scores, ids, median, average
