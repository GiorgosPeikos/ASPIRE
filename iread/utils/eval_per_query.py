import numpy as np
import streamlit
import streamlit as st
import statistics
import ir_measures
from utils.eval_single_exp import *
from utils.eval_multiple_exp import *
from collections import defaultdict, Counter
from utils.plots import *
from utils.eval_core import *
from typing import Dict, Any, List


@st.cache_data
def calculate_median_metrics(results_per_run: Dict[str, Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Calculate median metric values across N-1 experiments for each query.

    This function computes the median value for each metric and query,
    considering all experiments except the current one.

    Args:
    results_per_run (Dict[str, Dict[str, Dict[str, List[float]]]]): A nested dictionary
        containing evaluation results for each experiment, metric, and query.

    Returns:
    Dict[str, Dict[str, List[float]]]: A dictionary with median values for each experiment and metric.
    """
    median_results = {}
    experiments = list(results_per_run.keys())
    metrics = list(results_per_run[experiments[0]].keys())
    num_queries = len(results_per_run[experiments[0]][metrics[0]][metrics[0]])

    for current_exp in experiments:
        median_results[current_exp] = {}
        for metric in metrics:
            other_exps = [exp for exp in experiments if exp != current_exp]
            median_values = []

            for query_idx in range(num_queries):
                other_values = [results_per_run[exp][metric][metric][query_idx] for exp in other_exps]
                median_values.append(statistics.median(other_values))

            median_results[current_exp][f"median_{metric}"] = median_values

    return median_results


@st.fragment
@st.cache_data
def per_query_evaluation(qrel, runs, metric_list, relevance_threshold, selected_cutoff, baseline_run, threshold_value):
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

        # Call the plot function per measure scores
        plot_performance_measures_per_q(results_per_run)

        return results_per_run

    elif baseline_run:
        for run_name, run_data in runs.items():
            experiment = get_experiment_name(run_name, baseline_run)
            results_per_run[experiment] = {}
            for parsed_metric in parsed_metrics:
                results_per_run[experiment][str(parsed_metric)] = calculate_evaluation(parsed_metric, qrel, run_data)

        # Call the plot function for differences to the baseline
        plot_performance_difference(results_per_run)
        return results_per_run

    # This is the default run. So the threshold value needs to be equal to the median of the N-1 values.
    elif threshold_value == 0.0:
        for run_name, run_data in runs.items():
            experiment = get_experiment_name(run_name, baseline_run)
            results_per_run[experiment] = {}
            for parsed_metric in parsed_metrics:
                results_per_run[experiment][str(parsed_metric)] = calculate_evaluation(parsed_metric, qrel, run_data)

        # Calculate median metrics
        median_results = calculate_median_metrics(results_per_run)

        # Merge original results with median results
        for experiment in results_per_run:
            results_per_run[experiment].update(median_results[experiment])

        plot_performance_and_median_per_experiment(results_per_run)
        return results_per_run

    # The user has selected a float value.
    elif threshold_value != 0.0:
        for run_name, run_data in runs.items():
            experiment = get_experiment_name(run_name, baseline_run)
            results_per_run[experiment] = {}
            for parsed_metric in parsed_metrics:
                results_per_run[experiment][str(parsed_metric)] = calculate_evaluation(parsed_metric, qrel, run_data)

        return results_per_run

    elif threshold_value and baseline_run:
        # call function that estimate the performance to this threshold, default value is the median performance of all runs per query.
        return


@st.fragment
@st.cache_data
def find_same_performance_queries(data, runs, measure, max_queries):
    same_performance = []
    for query_id in range(max_queries):
        values = [data[run][measure][measure][query_id] if query_id < len(data[run][measure][measure]) else None for run in runs]
        if all(v is not None and abs(v - values[0]) < 1e-6 for v in values):  # Using small epsilon for float comparison
            same_performance.append(query_id + 1)  # +1 to make it 1-indexed
    return same_performance


@st.fragment
@st.cache_data
def find_large_gap_queries(data, runs, measure, max_queries):
    large_gaps = []
    all_values = []

    # Collect all values for this measure across all runs and queries
    for run in runs:
        all_values.extend(data[run][measure][measure])

    # Calculate Q1, Q3, and IQR
    q1 = np.percentile(all_values, 25)
    q3 = np.percentile(all_values, 75)
    iqr = q3 - q1

    # Define the threshold as 1.5 times the IQR
    threshold = 1.5 * iqr

    for query_id in range(max_queries):
        values = [data[run][measure][measure][query_id] if query_id < len(data[run][measure][measure]) else None for run in runs]
        values = [v for v in values if v is not None]
        if values and max(values) - min(values) > threshold:
            large_gaps.append((query_id + 1, min(values), max(values), threshold))

    return large_gaps, threshold


@st.fragment
@st.cache_data
def analyze_performance_perq(data):
    # Extract measures and runs
    eval_measures = list(data[list(data.keys())[0]].keys())
    runs = list(data.keys())

    # Determine the maximum number of queries
    max_queries = max(len(data[run][measure][measure]) for run in runs for measure in eval_measures)

    results = {}
    for measure in eval_measures:
        same_performance = find_same_performance_queries(data, runs, measure, max_queries)
        large_gaps, threshold = find_large_gap_queries(data, runs, measure, max_queries)

        results[measure] = {
            "same_performance": same_performance,
            "large_gaps": large_gaps,
            "threshold": threshold
        }

    return results


@st.fragment
@st.cache_data
def analyze_performance_difference(results):
    # Extract measures and runs
    eval_measures = list(results[list(results.keys())[0]].keys())
    runs = list(results.keys())

    # Find the baseline run
    baseline_run = next(run for run in runs if "(Baseline)" in run)
    other_runs = [run for run in runs if run != baseline_run]

    analysis_results = {}

    for run in other_runs:
        analysis_results[run] = {}
        for measure in eval_measures:
            baseline_values = results[baseline_run][measure][measure]
            run_values = results[run][measure][measure]

            # Compare each query's performance
            diff_values = [run_val - baseline_val for run_val, baseline_val in zip(run_values, baseline_values)]
            improved_queries = [i + 1 for i, diff in enumerate(diff_values) if diff > 0]
            degraded_queries = [i + 1 for i, diff in enumerate(diff_values) if diff < 0]
            unchanged_queries = [i + 1 for i, diff in enumerate(diff_values) if diff == 0]

            # Calculate statistics
            avg_diff = np.mean(diff_values)
            median_diff = np.median(diff_values)
            std_diff = np.std(diff_values)

            # Calculate percentages
            total_queries = len(diff_values)
            pct_improved = (len(improved_queries) / total_queries) * 100
            pct_degraded = (len(degraded_queries) / total_queries) * 100
            pct_unchanged = (len(unchanged_queries) / total_queries) * 100

            analysis_results[run][measure] = {
                "improved_queries": improved_queries,
                "degraded_queries": degraded_queries,
                "unchanged_queries": unchanged_queries,
                "total_queries": total_queries,
                "pct_improved": pct_improved,
                "pct_degraded": pct_degraded,
                "pct_unchanged": pct_unchanged,
                "avg_diff": avg_diff,
                "median_diff": median_diff,
                "std_diff": std_diff
            }

    return analysis_results, baseline_run


@st.fragment
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
