import ir_measures

# from ir_measures import *
import streamlit as st
import statistics


@st.cache_data
def return_available_measures():
    freq_measures = [
        "AP@10",  # Average Precision
        "AP@100",  # Average Precision
        "AP@1000",  # Average Precision
        "nDCG@5",  # Normalized Discounted Cumulative Gain
        "nDCG@10",  # Normalized Discounted Cumulative Gain
        "nDCG@1000",  # Normalized Discounted Cumulative Gain
        "P@5",  # Precision
        "P@10",  # Precision
        "P@25",  # Precision
        "R@10",  # Recall
        "R@25",  # Recall
        "R@50",  # Recall
        "R@1000",  # Recall
        "RR@1000",  # Reciprocal Rank
        "Rprec",  # Precision at R
        "Bpref",  # Binary Preference
        "infAP",  # Inferred Average Precision
        "NumRelRet",  # Number of Relevant Retrieved Documents (Alias)
        "NumRel",
        "NumRet",
    ]

    custom_user = [
        "AP",  # Average Precision
        "nDCG",  # Normalized Discounted Cumulative Gain
        "P",  # Precision
        "R",  # Recall
        "RR",  # Reciprocal Rank
        "Rprec",  # Precision at R
        "Bpref",  # Binary Preference
        "infAP",  # Inferred Average Precision
        "NumRelRet",  # Number of Relevant Retrieved Documents (Alias)
        "NumRel",
        "NumRet",
    ]

    rest_measures = [
        # "Accuracy",
        # "alpha_DCG",
        # "alpha_nDCG",
        # "AP_IA",
        # "BPM",
        "Compat",
        # "ERR",
        # "ERR_IA",
        # "INSQ",
        # "INST",
        # "IPrec",
        "Judged",
        # "nERR_IA",
        # "nNRBP",
        # "NRBP",
        "NumQ",
        "RBP",
        # "SDCG",
        "SetAP",
        "SetF",
        "SetP",
        "SetR",
        # "StRecall",
        # "Success",
        # "α_DCG",
        # "α_nDCG"
    ]
    return freq_measures, rest_measures, custom_user


# The function gets the qrels and the run from the session and a selected metric from the user and returns the
# evaluation measure.
# Define a set of measures that support a relevance threshold based on the documentation
# https://ir-measur.es/en/latest/measures.html

# Define a set of measures that support relevance level e.g. (rel=2)
measures_with_rel_param = {
    "Accuracy",
    "alpha_DCG",
    "alpha_nDCG",
    "AP",
    "AP_IA",
    "Bpref",
    "ERR_IA",
    "infAP",
    "IPrec",
    "NERR8",
    "NERR9",
    "NERR10",
    "NERR11",
    "nERR_IA",
    "nNRBP",
    "NRBP",
    "P",
    "P_IA",
    "R",
    "RBP",
    "Rprec",
    "RR",
    "SetAP",
    "SetF",
    "SetP",
    "SetR",
    "NumRet",
    "NumRelRet",
}

# Define a set of measures that support a cutoff e.g. @10
measures_with_cutoff = {
    "Accuracy",
    "alpha_DCG",
    "alpha_nDCG",
    "AP",
    "AP_IA",
    "BPM",
    "ERR",
    "ERR_IA",
    "nDCG",
    "NERR8",
    "NERR9",
    "NERR10",
    "NERR11",
    "nERR_IA",
    "nNRBP",
    "NRBP",
    "P",
    "P_IA",
    "R",
    "RBP",
    "RR",
    "SDCG",
    "SetAP",
    "SetF",
    "SetP",
    "SetR",
    "StRecall",
    "Success",
}

# Define a set of measures that need to be return as int
measures_int = ["NumRelRet", "NumRel", "NumRet", "NumQ"]


def evaluate_single_run(qrels, run, metric, relevance_threshold):
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
    res_eval = ir_measures.calc_aggregate([parsed_metric], qrels, run)

    if base_metric in measures_int:
        return int(list(res_eval.values())[0])
    else:
        return round(list(res_eval.values())[0], 4)


def evaluate_single_run_custom(qrels, run, metric, cutoff, relevance_threshold):
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
    res_eval = ir_measures.calc_aggregate([parsed_metric], qrels, run)

    return parsed_metric, round(list(res_eval.values())[0], 4)


def per_topic_evaluation(qrels, run, metric, cutoff, relevance_threshold):
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
    res_eval = list(ir_measures.iter_calc([parsed_metric], qrels, run))

    return parsed_metric, res_eval


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
