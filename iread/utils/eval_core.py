import statistics
import statsmodels.stats.multitest
import ir_measures
import streamlit as st
from utils.eval_single_exp import *
import pandas as pd


@st.cache_data()
def evaluate_single_run(qrel, run, metric, relevance_threshold):
    # Check if metric contains '@' and split if it does
    if "@" in metric:
        base_metric, cutoff = metric.split("@")
        # Check if the measure supports a relevance threshold
        if base_metric in measures_with_rel_param:
            # Format the new measure string with the relevance threshold
            new_metric_str = f"{base_metric}(rel={relevance_threshold})@{cutoff}"
        else:
            new_metric_str = metric
    else:
        # For metrics without cutoff, use the base metric directly
        base_metric = metric
        if base_metric in measures_with_rel_param:
            new_metric_str = f"{base_metric}(rel={relevance_threshold})"
        else:
            new_metric_str = metric

    # Parse the new metric using ir_measures
    parsed_metric = ir_measures.parse_measure(new_metric_str)

    # Calculate the evaluation using the parsed metric
    res_eval = ir_measures.calc_aggregate([parsed_metric], qrel, run)

    if base_metric in measures_int:
        return int(list(res_eval.values())[0])
    else:
        return round(list(res_eval.values())[0], 4)


@st.cache_data()
def evaluate_single_run_custom(qrel, run, metric, cutoff, relevance_threshold):
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
    res_eval = ir_measures.calc_aggregate([parsed_metric], qrel, run)

    if parsed_metric in measures_int:
        return str(parsed_metric), int(list(res_eval.values())[0])
    else:
        return str(parsed_metric), round(list(res_eval.values())[0], 4)


@st.cache_data()
def get_relevant_and_unjudged(qrel, res) -> dict:
    """
    Merge 'res' and 'qrel' DataFrames on 'query_id' and 'doc_id'.
    For each unique 'query_id', determine:
    - The first rank positions for relevance thresholds (0, 1, 2).
    - The rank position where 'relevance' is NaN (first_unjudged).
    Returns a dictionary where keys are 'query_id' and values are dictionaries
    containing these rank positions.
    """

    # Merge res and qrel on query_id and doc_id
    merged_df = pd.merge(res, qrel, on=['query_id', 'doc_id'], how='left')

    # Initialize result dictionary
    ranking_per_relevance = {}

    # Extract all available relevance thresholds
    relevance_thresholds = qrel['relevance'].unique()

    # Iterate over each unique query_id in the DataFrame
    for query_id, group in merged_df.groupby('query_id'):
        # Sort group by score in descending order
        group_sorted = group.sort_values(by='score', ascending=False)

        # Initialize a dictionary for the current query_id
        query_result = {}

        # Find the first rank positions for relevance thresholds (0, 1, 2)
        for relevance_val in relevance_thresholds:
            # Find the first rank position for the current relevance value
            relevant_rows = group_sorted[group_sorted['relevance'] == relevance_val]
            first_rank = relevant_rows['rank'].iloc[0] if not relevant_rows.empty else f'{len(group_sorted)}'

            # Store the first rank position in the dictionary
            if not relevance_val == 0:
                query_result[f'Relevance_Label_{relevance_val}'] = first_rank
            else:
                query_result[f'Irrelevant_Document'] = first_rank

        # Find the rank position where 'relevance' is NaN (first_unjudged)
        nan_relevance_rows = group_sorted[group_sorted['relevance'].isna()]
        first_rank_nan_relevance = nan_relevance_rows['rank'].iloc[0] if not nan_relevance_rows.empty else f'{len(group_sorted)}'

        # Store the first rank position in the dictionary
        query_result['Unjudged_Document'] = first_rank_nan_relevance

        # Store the dictionary for the current query_id in the result dictionary
        ranking_per_relevance[query_id] = query_result

    return ranking_per_relevance


@st.cache_resource()
def generate_prec_recall_graphs(relevance_threshold, selected_qrel, selected_runs):
    """
    Generates a dictionary of precision-recall data.

    This function iterates over recall thresholds from 0.0 to 1.0 in increments of 0.1,
    evaluates the precision for each threshold using the `evaluate_single_run` function,
    and stores the results in a dictionary.

    Parameters:
    - relevance_threshold (float): The relevance threshold to use for evaluation.
    - selected_qrel (any): The selected qrel data.
    - selected_runs (any): The selected runs data.

    Returns:
    - dict: A dictionary with keys in the format 'IPrec(rel={relevance_threshold})@{cutoff}'
      and values representing the precision at those recall thresholds.
    """
    prec_recall_graphs = {}

    for i in range(0, 11):
        cutoff = float(i / 10)
        key = f"IPrec(rel={relevance_threshold})@{cutoff}"
        prec_recall_graphs[key] = evaluate_single_run(
            selected_qrel, selected_runs, key, relevance_threshold
        )

    return prec_recall_graphs


def initialize_results():
    # Standard, extra, custom, and query results dictionaries are initialized
    if "results_standard" not in st.session_state:
        st.session_state["results_standard"] = {}
    if "results_extra" not in st.session_state:
        st.session_state["results_extra"] = {}
    if "results_custom" not in st.session_state:
        st.session_state["results_custom"] = {}
    if "results_query" not in st.session_state:
        st.session_state["results_query"] = {}
    if "saved_queries" not in st.session_state:
        st.session_state["saved_queries"] = {}
    if "selected_queries" not in st.session_state:
        st.session_state["selected_queries"] = {}
