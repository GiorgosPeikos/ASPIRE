import streamlit as st
import os
st.set_page_config(layout="wide", initial_sidebar_state="collapsed", )
from utils.data_handler import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css

# Load custom CSS
load_css("css/styles.css")

# Title for the page
st.markdown("""<div style="text-align: center;"><h1>Retrieval Evaluation Report based on the Retrieved Documents<h1></div>""", unsafe_allow_html=True)

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
    st.session_state.me_selected_qrels = load_qrel_data(os.path.join(qrels_dir, qrels_file))
    st.session_state.me_max_relevance = st.session_state.me_selected_qrels["relevance"].max()

    # Load selected runs data
    st.session_state.me_selected_runs = {}
    for run_file in selected_runs_files:
        st.session_state.me_selected_runs[run_file] = load_run_data(os.path.join(runs_dir, run_file))

    if st.session_state.me_selected_runs:
        st.markdown(
            f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{", ".join(selected_runs_files).replace('.txt', '').replace('.csv', '')}</span> experiments using the <span style="color:red;">{qrels_file}</span> qrels.</div>""",
            unsafe_allow_html=True)

    if queries_file:
        st.session_state.me_selected_queries = load_query_data(os.path.join(queries_dir, queries_file))
        st.markdown(
            f"""<div style="text-align: center;">This experiment is associated with the <span style="color:red;">{queries_file}</span> queries.</div>""",
            unsafe_allow_html=True)

st.divider()

# Documents per query assessed.

# Documents that have relevance assessment for more than one query.

# Documents that have been retrieved by all systems.

# Documents that have been retrieved by 1,2,3,5 systems.

# The box like plot for the top 10 positions showing how the documents are ranked.
