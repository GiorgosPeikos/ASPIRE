import streamlit as st
import numpy as np
from collections import Counter


# Helper function to get judgements for a specific label
def get_label_judgements(data, queries, label):
    # For 'Relevance_Label_0', return irrelevant judgements
    if label == 'Relevance_Label_0':
        return {q: data[q]['irrelevant'] for q in queries}
    # For other labels, return relevant judgements
    else:
        return {q: data[q]['relevant'][label] for q in queries}


def classify_queries(values, n_hard=5, n_easy=5):
    sorted_queries = sorted(values.items(), key=lambda x: x[1])
    hard_queries = [q for q, _ in sorted_queries[:n_hard]]
    easy_queries = [q for q, _ in sorted_queries[-n_easy:]]
    values_list = list(values.values())
    return {
        'hard': hard_queries,
        'easy': easy_queries,
        'median': np.median(values_list),
        'mean': np.mean(values_list),
        'min': min(values_list),
        'max': max(values_list)
    }


def compare_relevance_labels(data, queries, relevance_labels):
    results = {}
    irrelevant_label = 'Relevance_Label_0'

    for label in relevance_labels:
        if label == irrelevant_label:
            continue

        label_judgements = get_label_judgements(data, queries, label)
        irrelevant_judgements = get_label_judgements(data, queries, irrelevant_label)

        easy_queries = []
        hard_queries = []
        min_query = max_query = queries[0]
        min_count = max_count = label_judgements[queries[0]]

        for query in queries:
            label_count = label_judgements[query]
            irrelevant_count = irrelevant_judgements[query]

            if label_count >= irrelevant_count:
                easy_queries.append(query)
            elif label_count < irrelevant_count / 2:  # Arbitrary threshold for "very few"
                hard_queries.append(query)

            if label_count < min_count:
                min_count = label_count
                min_query = query
            if label_count > max_count:
                max_count = label_count
                max_query = query

        results[label] = {
            'easy_queries': easy_queries,
            'hard_queries': hard_queries,
            'min_query': min_query,
            'max_query': max_query
        }

    # Combine all relevant labels
    combined_judgements = {q: sum(get_label_judgements(data, [q], label)[q] for label in relevance_labels if label != irrelevant_label)
                           for q in queries}
    irrelevant_judgements = get_label_judgements(data, queries, irrelevant_label)

    easy_queries = []
    hard_queries = []
    min_query = max_query = queries[0]
    min_count = max_count = combined_judgements[queries[0]]

    for query in queries:
        combined_count = combined_judgements[query]
        irrelevant_count = irrelevant_judgements[query]

        if combined_count >= irrelevant_count:
            easy_queries.append(query)
        elif combined_count < irrelevant_count / 2:
            hard_queries.append(query)

        if combined_count < min_count:
            min_count = combined_count
            min_query = query
        if combined_count > max_count:
            max_count = combined_count
            max_query = query

    results['combined'] = {
        'easy_queries': easy_queries,
        'hard_queries': hard_queries,
        'min_query': min_query,
        'max_query': max_query
    }

    return results


@st.cache_data
def analyze_query_judgements(data):
    analysis_results = {}
    queries = list(data.keys())
    # Get all relevance labels, including 'Relevance_Label_0' for irrelevant judgements
    relevance_labels = ['Relevance_Label_0'] + list(data[queries[0]]['relevant'].keys())

    # 1. Sort queries based on the number of relevance judgements for each label
    sorted_queries = {}
    for label in relevance_labels:
        label_judgements = get_label_judgements(data, queries, label)
        # Sort queries by judgement count in descending order
        sorted_queries[label] = sorted(label_judgements.items(), key=lambda x: x[1], reverse=True)
    analysis_results['sorted_queries'] = sorted_queries

    # 2. Calculate statistics for each relevance label
    for label in relevance_labels:
        values = list(get_label_judgements(data, queries, label).values())
        analysis_results[f'{label}_stats'] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values)
        }

    # 3. Identify hard, normal, and very easy queries for each label and overall
    analysis_results['query_difficulty'] = {}
    for label in relevance_labels:
        values = get_label_judgements(data, queries, label)
        difficulty_classification = classify_queries(values)
        analysis_results['query_difficulty'][label] = difficulty_classification

    # 4. Calculate overall relevance statistics
    analysis_results['overall_stats'] = {}
    for label in relevance_labels:
        # Sum up all judgements for each label
        total = sum(get_label_judgements(data, queries, label).values())
        analysis_results['overall_stats'][label] = total

    # Calculate percentages for each label
    total_judgements = sum(analysis_results['overall_stats'].values())
    for label in relevance_labels:
        analysis_results['overall_stats'][f'{label}_percentage'] = (analysis_results['overall_stats'][label] / total_judgements) * 100

    # Add new analysis
    analysis_results['label_comparison'] = compare_relevance_labels(data, queries, relevance_labels)

    return analysis_results
