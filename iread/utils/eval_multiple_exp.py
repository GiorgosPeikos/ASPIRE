import numpy as np
from scipy import stats
import statsmodels.stats.multitest
import ir_measures
from ir_measures import *
from utils.run_analysis import *
import os
from collections import defaultdict
import pandas as pd


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
    if "@" in metric:
        base_metric, cutoff = metric.split("@")
    else:
        base_metric = metric

    if base_metric in measures_with_rel_param:  # Assuming measures_with_rel_param is defined elsewhere
        if cutoff is not None:
            new_metric_str = f"{base_metric}(rel={relevance_threshold})@{cutoff}"
        else:
            new_metric_str = f"{base_metric}(rel={relevance_threshold})"
    else:
        new_metric_str = metric

    # Parse the metric using ir_measures (assuming parse_measure is a function from ir_measures)
    parsed_metric = ir_measures.parse_measure(new_metric_str)
    return parsed_metric


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


def create_results_dataframe(results_dict):
    # Initialize lists to store data for DataFrame construction
    data = []
    index = []

    # Iterate through each experiment in the results dictionary
    for experiment, metrics_dict in results_dict.items():
        index.append(experiment)
        experiment_results = {"Experiment": experiment}

        # Iterate through each metric for the current experiment
        for metric, metric_data in metrics_dict.items():
            experiment_results[metric + "_mean"] = metric_data["mean"]

            # Check if "reject" key exists and its value is True
            if "reject" in metric_data and metric_data["reject"] == True:
                experiment_results[metric + "_reject"] = '<span style="color:green;">True</span>'
            else:
                experiment_results[metric + "_reject"] = '<span style="color:red;">False</span>'

        # Append experiment_results to data list
        data.append(experiment_results)

    # Create DataFrame from data list with Experiment column as index
    df = pd.DataFrame(data).set_index('Experiment')

    return df


def evaluate_multiple_runs(qrel, runs, metric_list, relevance_threshold, baseline, correction_method='bonferroni', correction_alpha=0.05):
    """
    Evaluates multiple runs of information retrieval systems across various metrics,
    computes mean scores for each metric, and computes corrected p-values against a baseline run.

    Parameters:
    - qrel (dict): Dictionary containing ground truth relevance judgments.
    - runs (dict): Dictionary where keys are run names and values are dictionaries containing
      retrieval results for each run.
    - metric_list (list): List of metric strings to evaluate.
    - relevance_threshold (int or float): Relevance threshold to use in evaluation.
    - correction_method (str): Method for multiple comparison correction (e.g., 'bonferroni', 'fdr_bh').
    - baseline (str): Name of the baseline run for comparison.
    - correction_alpha (float): Alpha level for multiple comparison correction.

    Returns:
    - statistical_results (dict): Dictionary containing run names as keys and dictionaries of results as values.
                                  Each inner dictionary will have metric names as keys and a dictionary of results,
                                  including mean scores, p-values against the baseline run, corrected p-values, and
                                  rejection decisions.
    """
    results_per_run = {}
    parsed_metrics = []
    statistical_results = {}
    baseline_experiment = {}

    # Parse each metric in metric_list
    for metric in metric_list:
        parsed_metric = metric_parser(metric, relevance_threshold, None)
        parsed_metrics.append(parsed_metric)

    # Calculate evaluation results for the baseline
    for run_name, run_data in runs.items():
        if run_name == baseline:
            experiment = get_experiment_name(run_name, baseline)
            statistical_results[experiment] = {}
            for parsed_metric in parsed_metrics:
                baseline_experiment[str(parsed_metric)] = calculate_evaluation(parsed_metric, qrel, run_data)
                # Extract values from the dictionary to compute mean
                metric_values = list(baseline_experiment[str(parsed_metric)].values())
                statistical_results[experiment][str(parsed_metric)] = {
                    "mean": np.mean(metric_values)
                }

    # Collect p-values for correction
    all_p_values = []

    # Calculate evaluation results for each other experiment and collect p-values
    for run_name, run_data in runs.items():
        if run_name != baseline:
            experiment = get_experiment_name(run_name, baseline)
            statistical_results[experiment] = {}
            for parsed_metric in parsed_metrics:
                results_per_run[str(parsed_metric)] = calculate_evaluation(parsed_metric, qrel, run_data)
                # Extract values from the dictionary to compute mean
                metric_values = list(results_per_run[str(parsed_metric)].values())
                mean_value = np.mean(metric_values)

                baseline_values = list(baseline_experiment[str(parsed_metric)].values())

                # Check for valid data before t-test, baseline_values[0] to access its values
                if len(baseline_values[0]) > 1 and len(metric_values[0]) > 1:
                    t_statistic, p_value = stats.ttest_rel(baseline_values[0], metric_values[0])

                    # Check if the p_value or t_statistic are NaN
                    if np.isnan(t_statistic) or np.isnan(p_value):
                        print(f"NaN detected for metric {str(parsed_metric)} in run {experiment}")

                    # Collect p-value for correction
                    all_p_values.append(p_value)

                    # Update statistical results without corrected p-value for now
                    statistical_results[experiment][str(parsed_metric)] = {
                        "mean": mean_value,
                        "p_value": p_value,
                        "corrected_p_value": None,  # Placeholder for corrected p-value
                        "reject": None,  # Placeholder for rejection decision
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

    # Apply multiple comparison correction to all collected p-values
    if all_p_values:
        corrected_p_values = statsmodels.stats.multitest.multipletests(all_p_values, alpha=correction_alpha, method=correction_method)[1]

        # Update statistical results with corrected p-values and rejection decisions
        p_index = 0
        for run_name, run_data in runs.items():
            if run_name != baseline:
                experiment = get_experiment_name(run_name, baseline)
                for parsed_metric in parsed_metrics:
                    if statistical_results[experiment][str(parsed_metric)]["p_value"] is not None:
                        statistical_results[experiment][str(parsed_metric)]["corrected_p_value"] = corrected_p_values[p_index]
                        statistical_results[experiment][str(parsed_metric)]["reject"] = corrected_p_values[p_index] < correction_alpha
                        p_index += 1

    results = create_results_dataframe(statistical_results)

    return results, statistical_results
