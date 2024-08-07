import numpy as np
import os
import streamlit as st
from utils.eval_core import *


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
