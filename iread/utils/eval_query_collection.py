import streamlit as st
import numpy as np
import pandas as pd
import ast
from utils.eval_single_exp import get_experiment_name


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


def get_query_rel_judgements(qrels):
    # Group by query_id and relevance, count occurrences
    relevance_counts = qrels.groupby(['query_id', 'relevance']).size().unstack(fill_value=0)

    # Rename columns
    relevance_counts.columns = ['Irrelevant' if col == 0 else f'Relevance_Label_{col}' for col in relevance_counts.columns]

    # Prepare results dictionary
    results = {}
    for i, query_id in enumerate(relevance_counts.index, start=1):  # Start enumeration from 1
        results[i] = {  # Use i (1-based) as the key instead of query_id
            "irrelevant": int(relevance_counts.loc[query_id, "Irrelevant"]),
            "relevant": {}
        }
        for column in relevance_counts.columns:
            if column != "Irrelevant":
                results[i]["relevant"][column] = int(relevance_counts.loc[query_id, column])

    return relevance_counts, results


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


@st.cache_data
def find_multi_query_docs(qrels):
    # Count the number of unique queries per document
    doc_query_count = qrels.groupby('doc_id')['query_id'].nunique()

    # Filter for documents with more than one query
    multi_query_docs = doc_query_count[doc_query_count > 1]

    # Sort by number of queries, descending
    multi_query_docs = multi_query_docs.sort_values(ascending=False)

    # Create a DataFrame with document IDs and their query counts
    result = pd.DataFrame({
        'doc_id': multi_query_docs.index,
        'query_count': multi_query_docs.values
    })

    # Add relevance information
    relevance_info = qrels.groupby('doc_id').apply(lambda x: x.groupby('query_id')['relevance'].first().to_dict())
    result['relevance_judgments'] = result['doc_id'].map(relevance_info)

    return result


@st.cache_data
def display_further_details_multi_query_docs(multi_query_docs, num_docs):
    # Dynamically get unique relevance labels from the data
    all_relevance = [rel for doc_rel in multi_query_docs['relevance_judgments'] for rel in doc_rel.values()]
    unique_relevance = sorted(set(all_relevance))

    avg_queries = multi_query_docs['query_count'].mean()
    max_queries = multi_query_docs['query_count'].max()
    top_docs = multi_query_docs.head(num_docs)

    st.write(f"Average number of relevance judgements across queries per document: <span style='color:red;'>{avg_queries:.2f}</span>", unsafe_allow_html=True)
    st.write(f"Maximum number of relevance judgements for a single document: <span style='color:red;'>{max_queries}</span>", unsafe_allow_html=True)

    st.write(f"Note that if a document has relevance judgements equal to the total number of queries in the collection it means that the document is somehow related to all the queries in the "
             f"collection. For example, if the selected collection has 75 queries, then if a document with id XYZ has Rel.Judgments across queries=75, it means that the document's relevance has been "
             f"assessed w.r.t. "
             "to all queries.", unsafe_allow_html=True)

    # Display a table for the top 5 documents
    st.markdown(f"##### Top {num_docs} Documents ranked by their relevance assessments across queries")

    top_docs_info = []

    for _, row in top_docs.iterrows():
        doc_id = row['doc_id']
        query_counts = row['query_count']
        relevance_counts = pd.Series(row['relevance_judgments'].values()).value_counts().to_dict()

        doc_info = {
            'Document ID': doc_id,
            'Rel.Judgments across queries': query_counts
        }

        for rel_label in unique_relevance:
            doc_info[f'Relevance {rel_label}'] = relevance_counts.get(rel_label, 0)

        top_docs_info.append(doc_info)

    top_docs_df = pd.DataFrame(top_docs_info)
    st.dataframe(top_docs_df, hide_index=True)


@st.cache_data
def find_ranked_pos_of_multi_query_docs(multi_query_docs, num_docs, experiments, qrels):
    # Limit to the top `num_docs` documents from the multi-query document dataframe
    docs_multi_asments = multi_query_docs.head(num_docs)

    # Ensure all identifiers are treated as strings
    docs_multi_asments['doc_id'] = docs_multi_asments['doc_id'].astype(str)

    # Extract doc_ids from docs_multi_asments and limit to 20
    doc_ids = docs_multi_asments['doc_id'].unique()[:50]  # Limit to a maximum of 20 documents

    # Notify the user that only the top 20 documents are shown
    st.write(f"**Note:** Only the top 50 document IDs are presented in this analysis.")

    # Track missing doc_ids
    missing_docs = set(doc_ids)

    # Iterate through each experiment's data
    for run, run_data in experiments.items():
        experiment = get_experiment_name(run, None)
        st.write(f"""<center><h5>Analysis of the <span style="color:red;">{experiment}</span> Experiment</h5></center>""", unsafe_allow_html=True)

        run_data['doc_id'] = run_data['doc_id'].astype(str)
        run_data['query_id'] = run_data['query_id'].astype(str)

        # Filter run_data for the relevant doc_ids
        filtered_run_data = run_data[run_data['doc_id'].isin(doc_ids)]

        # Merge filtered_run_data with qrels to get relevance labels
        merged_data = pd.merge(filtered_run_data, qrels, on=['query_id', 'doc_id'], how='left')
        merged_data.rename(columns={'relevance': 'relevance_label'}, inplace=True)

        # Ensure relevance_label is always an integer and replace NaN with 'Unjudged'
        merged_data['relevance_label'] = merged_data['relevance_label'].fillna(-1).astype(int).replace(-1, 'Unjudged')

        # Start a new column layout with 3 columns
        columns = st.columns(3)

        # Track the current column index
        column_index = 0

        # For each document, prepare and display tables
        for doc_id in doc_ids:
            # Filter merged_data for the current doc_id
            doc_data = merged_data[merged_data['doc_id'] == doc_id]

            if not doc_data.empty:
                # Prepare a DataFrame for the current doc_id
                combined_data = doc_data[['query_id', 'rank', 'score', 'relevance_label']]

                # Display the table in the current column
                with columns[column_index]:
                    st.write(f"""<center>Per query ranking details of Document: <span style="color:red;">{doc_id}</span></center>""", unsafe_allow_html=True)
                    st.dataframe(combined_data, hide_index=True)

                # Update column_index and reset if it reaches 3
                column_index = (column_index + 1) % 3

                # Remove from missing_docs if found
                missing_docs.discard(doc_id)

        # Show missing doc_ids
        if missing_docs:
            st.write(f"**The following documents have not been retrieved by the evaluated experiment:** {', '.join(missing_docs)}")
        else:
            st.write(f"**All examined documents have been retrieved by the evaluated experiment.**")
