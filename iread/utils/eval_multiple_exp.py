import scipy.stats
import statsmodels.stats.multitest
import pandas as pd
import numpy as np
from ir_measures import parse_measure, evaluator  # Assuming these are functions/classes from ir_measures
from utils.run_analysis import *

def evaluate_multiple_runs(qrel, runs, metric_list, relevance_threshold, correction_method, baseline):
    # Initialize an empty dictionary to store results for each run and metric
    results = {run: {} for run in runs}

    # Initialize empty list to store parsed metrics
    parsed_metrics = []

    # Process each metric in metric_list to handle relevance thresholds if needed
    for metric in metric_list:
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

        # Parse the metric using ir_measures (assuming parse_measure is a function from ir_measures)
        parsed_metric = parse_measure(new_metric_str)
        parsed_metrics.append(parsed_metric)

    # Initialize evaluator for the current run
    evaluator_instance = evaluator(parsed_metrics, qrel)

    # Evaluate each run with all metrics and store results
    for run_name, run_data in runs.items():
        print(f"Evaluating run: {run_name}")  # Debug print

        # Calculate aggregate for all metrics in one pass
        res_eval = evaluator_instance.calc_aggregate(run_data)

        # Store results in the dictionary
        results[run_name] = res_eval

    # Evaluate baseline run with all metrics and store results if baseline is provided
    if baseline and baseline in runs:
        print(f"Evaluating baseline run: {baseline}")  # Debug print

        # Initialize evaluator for the baseline run
        baseline_evaluator = evaluator(parsed_metrics, qrel)

        # Calculate aggregate for all metrics in one pass
        baseline_results = baseline_evaluator.calc_aggregate(runs[baseline])

        # Store results in the dictionary under 'baseline'
        results['baseline'] = baseline_results

    # Perform statistical tests (paired t-tests) between each run and the baseline
    ttest_results = {}

    for run_name, run_data in runs.items():
        if run_name != 'baseline' and 'baseline' in results:
            print(f"Performing t-test between {run_name} and baseline")  # Debug print

            # Extract scores for the metrics in metric_list
            baseline_scores = [results['baseline'][metric] for metric in parsed_metrics]
            current_scores = [results[run_name][metric] for metric in parsed_metrics]

            # Perform paired t-test
            t_stat, p_value = scipy.stats.ttest_rel(baseline_scores, current_scores)
            ttest_results[run_name] = (t_stat, p_value)

    # Apply multiple testing correction if specified
    if correction_method is not None:
        p_values = [p_value for _, (_, p_value) in ttest_results.items()]
        _, corrected_p_values, _, _ = statsmodels.stats.multitest.multipletests(p_values, alpha=0.05, method=correction_method)

        # Update ttest_results with corrected p-values
        index = 0
        for key, (t_stat, p_value) in ttest_results.items():
            if key in ttest_results:  # Ensure key exists
                if index < len(corrected_p_values):  # Check bounds of corrected_p_values
                    ttest_results[key] = (t_stat, p_value, corrected_p_values[index])
                    index += 1
                else:
                    ttest_results[key] = (t_stat, p_value, np.nan)  # Handle case where corrected p-value is missing

    # Prepare the final DataFrame structure
    final_results = []

    # Iterate through each run and construct the final structure
    for run_name, eval_results in results.items():
        if run_name == 'baseline':
            continue

        # Prepare a dictionary for this run
        run_dict = {
            'Experiment': run_name
        }

        # Iterate through each evaluation measure and add to run_dict
        for metric, score in eval_results.items():
            run_dict[f"{metric}"] = np.mean(score)
            if run_name in ttest_results:
                _, p_value, corrected_p_value = ttest_results[run_name]
                run_dict[f"{metric}_p-value"] = p_value
                run_dict[f"{metric}_p-value-corrected"] = corrected_p_value

        # Add this run's dictionary to the final_results list
        final_results.append(run_dict)

    # Convert final_results to DataFrame
    ttest_df = pd.DataFrame(final_results)

    return ttest_df
