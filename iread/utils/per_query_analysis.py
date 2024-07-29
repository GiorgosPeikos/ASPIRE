import numpy as np
import streamlit
import streamlit as st

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
