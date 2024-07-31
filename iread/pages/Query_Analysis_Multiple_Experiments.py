import os
import math
import pandas as pd
import numpy as np
import streamlit as st
from utils.data_handler import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.eval_multiple_exp import evaluate_multiple_runs_custom, get_doc_intersection, get_docs_retrieved_by_all_systems
from utils.eval_core import evaluate_single_run, return_available_measures, get_relevant_and_unjudged, generate_prec_recall_graphs
from utils.plots import dist_of_retrieved_docs, plot_precision_recall_curve

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Load custom CSS
load_css("css/styles.css")

# Title for the page
st.markdown("""<div style="text-align: center;"><h1>Query-based Analysis Across Multiple Experiments<h1></div>""", unsafe_allow_html=True)

# Container for loading data
with st.container():
    st.markdown("<h3>Loading Queries, Qrels, Runs for Analysis</h3>", unsafe_allow_html=True)
    columns = st.columns([1, 1, 3])

    # Select Queries
    with columns[0]:
        queries_dir = "../retrieval_experiments/queries"
        queries_file = st.selectbox("Select Queries", os.listdir(queries_dir) if os.path.exists(queries_dir) else [])

    # Select Qrels
    with columns[1]:
        qrels_dir = "../retrieval_experiments/qrels"
        qrels_file = st.selectbox("Select Qrels", os.listdir(qrels_dir) if os.path.exists(qrels_dir) else [])

    # Select Retrieval Runs (multiselect)
    with columns[2]:
        runs_dir = "../retrieval_experiments/retrieval_runs"
        selected_runs_files = st.multiselect("Select Retrieval Runs", os.listdir(runs_dir) if os.path.exists(runs_dir) else [])

# Button to start evaluation
if st.button("Begin the Experimental Evaluation!", key='me_stButtonCenter'):

    if not qrels_file:
        st.write("Please select Qrels file to proceed.")
        st.stop()

    if not selected_runs_files:
        st.write("Please select at least one Retrieval Run file to proceed.")
        st.stop()

    # Load Qrels data
    st.session_state.qme_selected_qrels = load_qrel_data(os.path.join(qrels_dir, qrels_file))
    st.session_state.qme_max_relevance = st.session_state.qme_selected_qrels["relevance"].max()

    # Load selected runs data
    st.session_state.qme_selected_runs = {}
    for run_file in selected_runs_files:
        st.session_state.qme_selected_runs[run_file] = load_run_data(os.path.join(runs_dir, run_file))

    if st.session_state.qme_selected_runs:
        st.markdown(
            f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{", ".join(selected_runs_files).replace('.txt', '').replace('.csv', '')}</span> experiments using the <span style="color:red;">{qrels_file}</span> qrels.</div>""",
            unsafe_allow_html=True)

    if queries_file:
        st.session_state.qme_selected_queries = load_query_data(os.path.join(queries_dir, queries_file))
        st.markdown(
            f"""<div style="text-align: center;">This experiment is associated with the <span style="color:red;">{queries_file}</span> queries.</div>""",
            unsafe_allow_html=True)

# Functionality that allows to randomly select queries for analysis, size and queries.
if 'qme_selected_queries' in st.session_state and not st.session_state.qme_selected_queries.empty:
    if len(st.session_state.qme_selected_queries) > 100:
        with st.container():
            st.write("""<h3>Query Sampling - <span style="color:red;">Sampling Queries to Facilitate Experiment Analysis</span></h3>""", unsafe_allow_html=True)

            # Initialize random_state in session state if it doesn't exist
            if 'random_state' not in st.session_state:
                st.session_state.random_state = 42
            if 'random_size' not in st.session_state:
                st.session_state.random_size = 100

            st.write(f"""<div style="text-align: center;">  ⚠️ Note: Too many available queries (<span style="color:red;">{len(st.session_state.qme_selected_queries)}</span>).
            To enhance the following analysis, a random set will be used. Please select the following:</div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.session_state.random_size = st.number_input(
                    "Set Number of queries to be randomly selected (default: 49)",
                    min_value=1,
                    value=st.session_state.random_size,
                    max_value=len(st.session_state.qme_selected_queries),
                    step=1
                )
            with col2:
                st.session_state.random_state = st.number_input(
                    "Set random state for query selection (default: 42)",
                    min_value=1,
                    value=st.session_state.random_state,
                    max_value=100,
                    step=1
                )

            # Main content logic
            st.write(f"""<div style="text-align: center;"> A total of <span style="color:red;">{st.session_state.random_size}</span> random queries have been
            selected based on a random state equal to <span style="color:red;">{st.session_state.random_state}</span> and will be used for the upcoming analyses.</div>""", unsafe_allow_html=True)

            st.session_state.qme_selected_queries_random = st.session_state.qme_selected_queries.sample(n=st.session_state.random_size, random_state=st.session_state.random_state)

            st.write(f"""<div style="text-align: center;"> Number of randomly selected queries that would be used for analysis: <span style="color:red;"
                >{len(st.session_state.qme_selected_queries_random)}</span></div>""", unsafe_allow_html=True)

            query_ids = np.sort(st.session_state.qme_selected_queries_random.query_id.values)
            query_ids_str = ", ".join(map(str, query_ids))
            st.write(f"Selected Query (IDs): {query_ids_str}")

            st.divider()
    else:

        st.session_state.qme_selected_queries_random = st.session_state.qme_selected_queries
        st.write(f"""<div style="text-align: center;"> All <span style="color:red;">{len(st.session_state.qme_selected_queries_random)}</span> provided queries will be used for the 
        following analyses.</div>""", unsafe_allow_html=True)

        st.divider()


# Per query Measure Performance Plots
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Query-based Experimental Evaluation</span></h3>""", unsafe_allow_html=True)

    if 'qme_selected_qrels' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")
    else:
        st.write("ME")
