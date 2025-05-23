from collections import Counter, defaultdict

import numpy as np
import statsmodels.stats.multitest
from scipy import stats
from utils.eval_core import *
from utils.eval_single_exp import (create_results_table, get_experiment_name,
                                   metric_parser)


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
    query_ids = []
    if isinstance(parsed_metric, list):
        for metric in parsed_metric:
            for metric_result in ir_measures.iter_calc([metric], qrel, run_data):
                metric_name = str(metric_result.measure)
                metric_value = metric_result.value
                metric_scores[metric_name].append(metric_value)
                query_ids.append(metric_result.query_id)
    else:
        for metric_result in ir_measures.iter_calc([parsed_metric], qrel, run_data):
            metric_name = str(metric_result.measure)
            metric_value = metric_result.value
            metric_scores[metric_name].append(metric_value)
            query_ids.append(metric_result.query_id)

    # Add query_ids to the output
    for metric_name in metric_scores:
        metric_scores[metric_name] = {
            'values': metric_scores[metric_name],
            'query_ids': query_ids
        }

    return dict(metric_scores)


@st.cache_data
def evaluate_multiple_runs_custom(
    qrel,
    runs,
    metric_list,
    relevance_threshold,
    baseline,
    selected_cutoff,
    correction_method="bonferroni",
    correction_alpha=0.05,
):
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
                baseline_experiment[str(parsed_metric)] = calculate_evaluation(
                    parsed_metric, qrel, run_data
                )
                metric_values = baseline_experiment[str(parsed_metric)][str(parsed_metric)]['values']
                query_ids = baseline_experiment[str(parsed_metric)][str(parsed_metric)]['query_ids']
                statistical_results[experiment][str(parsed_metric)] = {
                    "mean": np.mean(metric_values),
                    "query_ids": query_ids
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

                results_per_run[str(parsed_metric)] = calculate_evaluation(
                    parsed_metric, qrel, run_data
                )
                metric_values = results_per_run[str(parsed_metric)][str(parsed_metric)]['values']
                query_ids = results_per_run[str(parsed_metric)][str(parsed_metric)]['query_ids']
                mean_value = np.mean(metric_values)

                baseline_values = baseline_experiment[str(parsed_metric)][str(parsed_metric)]['values']
                baseline_query_ids = baseline_experiment[str(parsed_metric)][str(parsed_metric)]['query_ids']

                # Ensure we're comparing the same queries
                common_query_ids = set(query_ids) & set(baseline_query_ids)
                metric_values = [v for v, qid in zip(metric_values, query_ids) if qid in common_query_ids]
                baseline_values = [v for v, qid in zip(baseline_values, baseline_query_ids) if qid in common_query_ids]

                if len(baseline_values) > 1 and len(metric_values) > 1:
                    t_statistic, p_value = stats.ttest_rel(
                        baseline_values, metric_values
                    )

                    if np.isnan(t_statistic) or np.isnan(p_value):
                        print(
                            f"NaN detected for metric {str(parsed_metric)} in run {experiment}"
                        )

                    metric_p_values.append(p_value)

                    statistical_results[experiment][str(parsed_metric)] = {
                        "mean": mean_value,
                        "p_value": p_value,
                        "corrected_p_value": None,  # Placeholder
                        "reject": None,  # Placeholder
                        "t_statistic": t_statistic,
                        "query_ids": list(common_query_ids)  # Store the common query IDs
                    }
                else:
                    print(
                        f"Insufficient data for t-test on metric {str(parsed_metric)} in run {experiment}"
                    )
                    statistical_results[experiment][str(parsed_metric)] = {
                        "mean": mean_value,
                        "p_value": float("nan"),
                        "corrected_p_value": float("nan"),
                        "reject": False,
                        "t_statistic": float("nan"),
                        "query_ids": list(common_query_ids)  # Store the common query IDs
                    }

        # Apply per-measure correction if needed
        if need_correction and metric_p_values:
            corrected_p_values = statsmodels.stats.multitest.multipletests(
                metric_p_values, alpha=correction_alpha, method=correction_method
            )[1]

            # Update statistical results with corrected p-values
            p_index = 0
            for run_name, run_data in runs.items():
                if run_name != baseline:
                    experiment = get_experiment_name(run_name, baseline)
                    if (
                        statistical_results[experiment][str(parsed_metric)]["p_value"]
                        is not None
                    ):
                        statistical_results[experiment][str(parsed_metric)][
                            "corrected_p_value"
                        ] = corrected_p_values[p_index]
                        statistical_results[experiment][str(parsed_metric)][
                            "reject"
                        ] = (corrected_p_values[p_index] < correction_alpha)
                        p_index += 1
        elif not need_correction:
            # If no correction needed, set corrected p-value same as original
            for run_name, run_data in runs.items():
                if run_name != baseline:
                    experiment = get_experiment_name(run_name, baseline)
                    statistical_results[experiment][str(parsed_metric)][
                        "corrected_p_value"
                    ] = statistical_results[experiment][str(parsed_metric)]["p_value"]
                    statistical_results[experiment][str(parsed_metric)]["reject"] = (
                        statistical_results[experiment][str(parsed_metric)]["p_value"]
                        < correction_alpha
                    )

    df, style_df = create_results_table(statistical_results)

    return df, style_df


@st.cache_data
def get_doc_intersection(runs, baseline, selected_cutoff):
    # Get unique query IDs and document IDs
    all_queries = set()
    all_docs = set()
    for run_data in runs.values():
        all_queries.update(run_data["query_id"])
        all_docs.update(run_data["doc_id"])

    query_map = {q: q for q in all_queries}  # Preserve string query IDs
    doc_map = {d: i for i, d in enumerate(all_docs)}

    n_queries = len(all_queries)
    n_docs = len(all_docs)
    n_runs = len(runs)

    # Create a 3D numpy array to represent the data
    data = np.zeros((n_runs, n_queries, n_docs), dtype=bool)

    # Fill the array
    for i, (run_name, run_data) in enumerate(runs.items()):
        for _, row in run_data.iterrows():
            if row["rank"] <= selected_cutoff:
                q_idx = list(query_map.keys()).index(row["query_id"])
                d_idx = doc_map[row["doc_id"]]
                data[i, q_idx, d_idx] = True

    # Compute intersections
    baseline_idx = list(runs.keys()).index(baseline)
    baseline_data = data[baseline_idx]
    intersections = (data & baseline_data).sum(axis=(1, 2))

    # Compute totals
    totals = np.minimum(data.sum(axis=2), selected_cutoff).sum(axis=1)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Intersected Documents": intersections,
            "Total Documents": totals,
        },
        index=list(runs.keys()),
    )

    df = df.drop(baseline)  # Remove baseline from results
    df["Intersection Percentage"] = (
        df["Intersected Documents"] / df["Total Documents"] * 100
    ).round(2)

    return df


@st.cache_data
def get_docs_retrieved_by_all_systems(runs, selected_cutoff, sample_size):
    # Get unique query IDs and document IDs
    all_queries = set()
    all_docs = set()
    for run_data in runs.values():
        all_queries.update(run_data["query_id"])
        all_docs.update(run_data["doc_id"])

    query_map = {q: q for q in all_queries}  # Preserve string query IDs
    doc_map = {d: i for i, d in enumerate(all_docs)}

    n_queries = len(all_queries)
    n_docs = len(all_docs)
    n_runs = len(runs)

    # Create a 3D numpy array to represent the data
    data = np.zeros((n_runs, n_queries, n_docs), dtype=bool)

    # Fill the array
    for i, (run_name, run_data) in enumerate(runs.items()):
        for _, row in run_data.iterrows():
            if row["rank"] <= selected_cutoff:
                q_idx = list(query_map.keys()).index(row["query_id"])
                d_idx = doc_map[row["doc_id"]]
                data[i, q_idx, d_idx] = True

    # Find documents retrieved by all systems
    docs_retrieved_by_all = data.all(axis=0)

    # Get the actual document IDs and query IDs
    retrieved_docs = []
    queries_with_retrieved_docs = set()
    for q_idx, d_idx in zip(*np.where(docs_retrieved_by_all)):
        query_id = list(query_map.keys())[q_idx]
        doc_id = list(doc_map.keys())[list(doc_map.values()).index(d_idx)]
        retrieved_docs.append(doc_id)
        queries_with_retrieved_docs.add(query_id)

    # Count occurrences of each document
    doc_counts = Counter(retrieved_docs)

    # Get the most frequent documents, up to sample_size
    most_frequent_docs = [doc for doc, _ in doc_counts.most_common(sample_size)]

    return (
        most_frequent_docs,
        len(queries_with_retrieved_docs),
        sorted(queries_with_retrieved_docs),
    )
