import pandas as pd
import streamlit as st
import numpy as np
import ir_measures
from ir_measures import *
import os


@st.cache_data
def return_available_measures():
    default_measures = [
        "AP@100",  # Average Precision
        "P@10",  # Precision
        "nDCG@10",  # Normalized Discounted Cumulative Gain
        "R@50",  # Recall
        "RR@1000",  # Reciprocal Rank
    ]

    overall_measures = [
        "NumQ",  # The total number of queries.
        "NumRet",  # Number of Retrieved Documents
        "NumRel",  # Number of Relevant Documents
        "NumRelRet",  # Number of Relevant Retrieved Documents (Alias)
    ]

    precision_measures = [
        "P@5",  # Number of Relevant Retrieved Documents (Alias)
        "P@10",
        "P@25",
        "P@50",
        "P@100",
        "Rprec"
    ]

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
        "Judged",
        "SetF",
        "SetP",
        "SetR",
        "SetAP",
        "Success",
        "Compat",
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
        # "RBP", - Error
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
    return freq_measures, rest_measures, custom_user, default_measures, overall_measures, precision_measures


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
    "StRecall",
    "Success",
}

# Define a set of measures that need to be return as int
measures_int = ["NumRelRet", "NumRel", "NumRet", "NumQ"]


def find_unjudged(run: pd.DataFrame, qrels: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    run["rank"] = run["rank"].astype(int)
    run = run[run["rank"] < cutoff]

    df = pd.merge(run, qrels, how="left", on=["doc_id", "query_id"])
    df.fillna(-1, inplace=True)

    return df[df["relevance"] == -1]


@st.cache_data  # Assuming this is a decorator for caching resources
def metric_parser(metric, relevance_threshold, cutoff):
    """
    Parses a metric string to handle relevance thresholds and cutoffs if specified.

    Parameters:
    - metric (str): The metric string to parse.
    - relevance_threshold (int or float): The relevance threshold to use in parsing.
    - cutoff (int or None): The cutoff value to use in parsing.

    Returns:
    - parsed_metric: The parsed metric object, parsed using ir_measures.parse_measure.
    """
    # Check if metric contains '@' and split if it does
    if "@" in metric:
        base_metric, existing_cutoff = metric.split("@")
        cutoff = existing_cutoff if cutoff is None else cutoff
    else:
        base_metric = metric

    # Check if the measure supports a relevance threshold and/or cutoff
    supports_rel = base_metric in measures_with_rel_param
    supports_cutoff = base_metric in measures_with_cutoff

    if supports_rel and supports_cutoff:
        new_metric_str = f"{base_metric}(rel={relevance_threshold})@{cutoff}"
    elif supports_rel:
        new_metric_str = f"{base_metric}(rel={relevance_threshold})"
    elif supports_cutoff:
        new_metric_str = f"{base_metric}@{cutoff}"
    else:
        new_metric_str = metric

    # Parse the metric using ir_measures
    parsed_metric = ir_measures.parse_measure(new_metric_str)
    return parsed_metric


def format_p_value(p_value):
    if p_value is None or np.isnan(p_value):
        return "-"
    elif p_value < 0.001:
        return "0.001"
    else:
        return f"{p_value:.3f}"


def get_experiment_name(run_name, baseline):
    """
    Extracts the experiment name from run_name by removing common file extensions
    and appends (Baseline) if the run_name matches the baseline.

    Parameters:
    - run_name (str): The name of the run or experiment.
    - baseline (str): The name of the baseline run.

    Returns:
    - experiment_name (str): Processed experiment name.
    """
    base_name, extension = os.path.splitext(run_name)

    # Append (Baseline) to experiment name if it's the baseline run
    experiment_name = base_name + " (Baseline)" if run_name == baseline else base_name

    return experiment_name


@st.cache_resource
def to_super(text):
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '.': '·', '-': '⁻'
    }
    return ''.join(superscript_map.get(char, char) for char in str(text))


@st.cache_resource
def to_sub(text):
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        '.': '.', '-': '₋'
    }
    return ''.join(subscript_map.get(char, char) for char in str(text))


@st.cache_data
def create_results_table(statistical_results):
    # Extract baseline and other system names
    baseline_system = next(iter(statistical_results.keys()))
    other_systems = [key for key in statistical_results.keys() if key != baseline_system]

    # Get metrics
    metrics = list(statistical_results[baseline_system].keys())

    # Create a DataFrame to store the results
    df = pd.DataFrame(columns=metrics, index=[baseline_system] + other_systems)

    # Create a DataFrame to store the styles
    style_df = pd.DataFrame('', columns=metrics, index=[baseline_system] + other_systems)

    # Fill the DataFrame
    for metric in metrics:
        max_mean = max(statistical_results[system][metric]['mean'] for system in [baseline_system] + other_systems)

        for system in [baseline_system] + other_systems:
            mean = statistical_results[system][metric]['mean']

            if system == baseline_system:
                # Baseline: just show the mean
                mean_str = f"{mean:.3f}"
                if mean == max_mean:
                    mean_str = '\u0332'.join(mean_str) + '\u0332'  # Add underlining
                df.loc[system, metric] = mean_str
            else:
                # Other system: show mean with p-values
                p_value = statistical_results[system][metric].get('p_value', np.nan)
                corrected_p_value = statistical_results[system][metric].get('corrected_p_value', np.nan)
                reject = statistical_results[system][metric].get('reject', False)

                formatted_p = format_p_value(p_value) if not np.isnan(p_value) else ""
                formatted_corrected_p = format_p_value(corrected_p_value) if not np.isnan(corrected_p_value) else ""

                mean_str = f"{mean:.3f}"
                if mean == max_mean:
                    mean_str = '\u0332' + '\u0332'.join(mean_str) + '\u0332'  # Add underlining

                # Create a stacked layout using box-drawing characters
                cell_value = f"{mean_str}  | \n{to_super(formatted_p)}\n\n{to_sub(formatted_corrected_p)}"

                df.loc[system, metric] = cell_value

                # Determine cell style
                style = []
                if reject and system != baseline_system:
                    style.append('background-color: lightgreen')
                style.append('color: black')
                if mean == max_mean:
                    style.append('font-weight: bold')
                    # Removed text-decoration: underline !important as we're now underlining in the string itself

                style_df.loc[system, metric] = '; '.join(style)

    return df, style_df
