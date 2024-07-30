import os
import math
import pandas as pd
import streamlit as st
from utils.data import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.eval_multiple_exp import evaluate_multiple_runs_custom, get_doc_intersection, get_docs_retrieved_by_all_systems
from utils.evaluation_measures import evaluate_single_run, return_available_measures, get_relevant_and_unjudged, generate_prec_recall_graphs
from utils.plots import dist_of_retrieved_docs, plot_precision_recall_curve

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Load custom CSS
load_css("css/styles.css")

# Title for the page
st.markdown("""<div style="text-align: center;"><h1>Retrieval Evaluation Report of Multiple Experiments<h1></div>""", unsafe_allow_html=True)

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

st.divider()

# Overall Retrieval Characteristics


