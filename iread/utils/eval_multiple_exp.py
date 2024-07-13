import numpy as np
import streamlit as st
from scipy import stats
import statsmodels.stats.multitest
import ir_measures
from ir_measures import *
from utils.run_analysis import *
from collections import defaultdict


def calculate_evaluation(parsed_metric, qrel, run_data):
    """
    Calculates the evaluation results for a single run based on parsed metrics.

    Parameters:
    - parsed_metric (object or list): Metric parsed using ir_measures.parse_measure or list of such metrics.
    - qrel (dict): Dictionary containing ground truth relevance judgments.
    - run_data (dict): Dictionary containing retrieval results for a single run.

    Returns:
    - evaluation_results (dict): Dictionary of evaluation scores computed for the run based on parsed_metric.
    """
    metric_scores = defaultdict(list)
    if isinstance(parsed_metric, list):
        for metric in parsed_metric:
            for metric_result in ir_measures.iter_calc([metric], qrel, run_data):
                metric_name = str(metric_result.measure)
                metric_value = metric_result.value
                metric_scores[metric_name].append(metric_value)
    else:
        for metric_result in ir_measures.iter_calc([parsed_metric], qrel, run_data):
            metric_name = str(metric_result.measure)
            metric_value = metric_result.value
            metric_scores[metric_name].append(metric_value)

    return dict(metric_scores)


def evaluate_multiple_runs_custom(qrel, runs, metric_list, relevance_threshold, baseline, selected_cutoff, correction_method='bonferroni', correction_alpha=0.05):
    results_per_run = {}
    parsed_metrics = []
    statistical_results = {}
    baseline_experiment = {}

    # Parse each metric in metric_list
    for metric in metric_list:
        parsed_metric = metric_parser(metric, relevance_threshold, selected_cutoff)
        parsed_metrics.append(parsed_metric)

    # Calculate evaluation results for the baseline
    for run_name, run_data in runs.items():
        if run_name == baseline:
            experiment = get_experiment_name(run_name, baseline)
            statistical_results[experiment] = {}
            for parsed_metric in parsed_metrics:
                baseline_experiment[str(parsed_metric)] = calculate_evaluation(parsed_metric, qrel, run_data)
                metric_values = list(baseline_experiment[str(parsed_metric)].values())
                statistical_results[experiment][str(parsed_metric)] = {
                    "mean": np.mean(metric_values)
                }

    # Determine if correction is needed (more than 2 experiments)
    need_correction = len(runs) > 2

    # Calculate evaluation results for each other experiment and collect p-values
    for parsed_metric in parsed_metrics:
        metric_p_values = []
        for run_name, run_data in runs.items():
            if run_name != baseline:
                experiment = get_experiment_name(run_name, baseline)
                if experiment not in statistical_results:
                    statistical_results[experiment] = {}

                results_per_run[str(parsed_metric)] = calculate_evaluation(parsed_metric, qrel, run_data)
                metric_values = list(results_per_run[str(parsed_metric)].values())
                mean_value = np.mean(metric_values)

                baseline_values = list(baseline_experiment[str(parsed_metric)].values())

                if len(baseline_values[0]) > 1 and len(metric_values[0]) > 1:
                    t_statistic, p_value = stats.ttest_rel(baseline_values[0], metric_values[0])

                    if np.isnan(t_statistic) or np.isnan(p_value):
                        print(f"NaN detected for metric {str(parsed_metric)} in run {experiment}")

                    metric_p_values.append(p_value)

                    statistical_results[experiment][str(parsed_metric)] = {
                        "mean": mean_value,
                        "p_value": p_value,
                        "corrected_p_value": None,  # Placeholder
                        "reject": None,  # Placeholder
                        "t_statistic": t_statistic
                    }
                else:
                    print(f"Insufficient data for t-test on metric {str(parsed_metric)} in run {experiment}")
                    statistical_results[experiment][str(parsed_metric)] = {
                        "mean": mean_value,
                        "p_value": float('nan'),
                        "corrected_p_value": float('nan'),
                        "reject": False,
                        "t_statistic": float('nan')
                    }

        # Apply per-measure correction if needed
        if need_correction and metric_p_values:
            corrected_p_values = statsmodels.stats.multitest.multipletests(metric_p_values, alpha=correction_alpha, method=correction_method)[1]

            # Update statistical results with corrected p-values
            p_index = 0
            for run_name, run_data in runs.items():
                if run_name != baseline:
                    experiment = get_experiment_name(run_name, baseline)
                    if statistical_results[experiment][str(parsed_metric)]["p_value"] is not None:
                        statistical_results[experiment][str(parsed_metric)]["corrected_p_value"] = corrected_p_values[p_index]
                        statistical_results[experiment][str(parsed_metric)]["reject"] = corrected_p_values[p_index] < correction_alpha
                        p_index += 1
        elif not need_correction:
            # If no correction needed, set corrected p-value same as original
            for run_name, run_data in runs.items():
                if run_name != baseline:
                    experiment = get_experiment_name(run_name, baseline)
                    statistical_results[experiment][str(parsed_metric)]["corrected_p_value"] = statistical_results[experiment][str(parsed_metric)]["p_value"]
                    statistical_results[experiment][str(parsed_metric)]["reject"] = statistical_results[experiment][str(parsed_metric)]["p_value"] < correction_alpha

    df, style_df = create_results_table(statistical_results)

    return df, style_df

