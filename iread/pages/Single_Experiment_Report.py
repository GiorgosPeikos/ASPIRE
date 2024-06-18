import os
import pandas as pd
import streamlit as st
from utils.data import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.evaluation_measures import evaluate_single_run, return_available_measures

load_css("css/styles.css")

# In this page, the user can load a single experiment and create an Evaluation Report for this experiment. Usage:
# This report can be useful if the user wants to discuss the results with others by sending them a single PDF file
st.markdown("""<div style="text-align: center;"><h1>Retrieval Evaluation Report<h1></div>""", unsafe_allow_html=True)

# This is the 1st container of the page, responsible for loading the data that will be used in the Analysis and reporting.
# It leverages the data loading functions implemented in the utils.data folder. By default, the system will display all
# available files. In case a file is missing appropriate messages are shown and the system stops.
with st.container():
    st.markdown("<h3>Loading Queries, Qrels, Runs for Analysis</h3>", unsafe_allow_html=True)

    # Create three columns for file selections
    columns = st.columns(3)

    # File selections for each folder
    with columns[0]:
        queries_file = st.selectbox("Select Queries", os.listdir("../retrieval_experiments/queries"))

    with columns[1]:
        qrels_file = st.selectbox("Select Qrels", os.listdir("../retrieval_experiments/qrels"))

    with columns[2]:
        st.session_state.runs_file = st.selectbox("Select Retrieval Run", os.listdir("../retrieval_experiments/retrieval_runs"))

# Example usage with caching
if st.button("Begin the Experimental Evaluation!", key='stButtonCenter'):
    # Load qrels file if selected
    if qrels_file:
        st.session_state.selected_qrels = load_qrel_data(os.path.join(f'../retrieval_experiments/qrels/{qrels_file}'))
        st.session_state.max_relevance = st.session_state['selected_qrels']["relevance"].max()  # Get maximum relevance level
    else:
        st.write("Please select Qrels file to proceed.")
        st.stop()  # Stop execution if qrels file is not selected

    # Load runs file if selected
    if st.session_state.runs_file:
        st.session_state.selected_runs = load_run_data(os.path.join(f'../retrieval_experiments/retrieval_runs/{st.session_state.runs_file}'))
    else:
        st.write("Please select Retrieval Run file to proceed.")
        st.stop()  # Stop execution if runs file is not selected

    # Display messages and store data in session state if qrels and runs files are selected
    st.markdown(
        f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{st.session_state.runs_file.replace('.txt', '')}</span> experiment using the <span style="color:red;">{qrels_file}
        </span> qrels.</div>""",
        unsafe_allow_html=True,
    )

    # Load query file if selected
    if queries_file:
        st.session_state.selected_queries = load_query_data(os.path.join(f'../retrieval_experiments/queries/{queries_file}'))

        st.markdown(
            f"""<div style="text-align: center;">This experiment is associated with the <span style="color:red;">{queries_file}</span> queries.</div>""",
            unsafe_allow_html=True,
        )

# Proceed to analysis, after the user has selected experiment and qrels
st.divider()

# This is the 2nd container of the page that presents the overall characteristics of the evaluated retrieval run.
# Presents: Number of topics, Number of retrieved documents, Total Relevant documents, Total relevant retrieved
# documents
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Overall Retrieval 
    Characteristics</span></h3>""", unsafe_allow_html=True)

    if st.session_state.runs_file:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")

    columns = st.columns(2)
    # Process and display a predefined set of evaluation measures based on the current threshold
    _, _, _, _, overall_measures, precision_measures = return_available_measures()

    with columns[0]:
        if not st.session_state.selected_qrels.empty:
            overall_measures_results = {}
            for measure_name in overall_measures:
                overall_measures_results[measure_name] = evaluate_single_run(
                    st.session_state.selected_qrels, st.session_state.selected_runs, measure_name, 2
                )

        # Display the results in a dataframe
        st.dataframe(pd.DataFrame([overall_measures_results], index=[st.session_state.runs_file.replace('.txt', '')]).transpose())

    with columns[1]:
        if not st.session_state.selected_qrels.empty:
            precision_measures_results = {}
            for measure_name in precision_measures:
                precision_measures_results[measure_name] = evaluate_single_run(
                    st.session_state.selected_qrels, st.session_state.selected_runs, measure_name, 1
                )
        # Display the results in a dataframe
        st.dataframe(pd.DataFrame([precision_measures_results], index=[st.session_state.runs_file.replace('.txt', '')]).transpose())


st.divider()

# This is the 3rd container of the page that estimates the main evaluation measures for the provided experiment and
# presents them as tables.
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Common Evaluation Measures</span></h3>""", unsafe_allow_html=True)

    # Process and display a predefined set of evaluation measures based on the current threshold
    _, _, _, default_measures, _, _ = return_available_measures()

    if st.session_state.selected_qrels.empty:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")

    # Create a slider and update the session state when the value changes
    st.session_state.relevance_threshold = st.slider(
        "Select from the Available Relevance Thresholds (Slide)",
        min_value=1,
        max_value=2,
        value=1,
    )

    # Initialize the session state for prev_relevance_threshold if it doesn't exist
    # Relevance label as 1 is a common convention in IR
    if 'prev_relevance_threshold' not in st.session_state:
        st.session_state.prev_relevance_threshold = 1
    # Update the session state with the current slider value
    if st.session_state.relevance_threshold != st.session_state.prev_relevance_threshold:
        st.session_state.prev_relevance_threshold = st.session_state.relevance_threshold

    if not st.session_state.selected_qrels.empty:
        freq_measures_results = {}
        for measure_name in default_measures:
            freq_measures_results[measure_name] = evaluate_single_run(
                st.session_state.selected_qrels, st.session_state.selected_runs, measure_name, st.session_state.relevance_threshold
            )

    # Display the results in a dataframe
    st.dataframe(pd.DataFrame([freq_measures_results], index=[st.session_state.runs_file.replace('.txt', '')]))

st.divider()

# This is the 4th container of the page that contains a graph that depicts the distribution of relevant documents
# retrieved based on their position. For instance, for how many queries the 1st document retrieved is relevant?
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Positional Distribution of Relevant Retrieved 
    Documents</span></h3>""", unsafe_allow_html=True)

st.divider()

# This is the 5th container of the page that presents the precision-recall curve.
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Precision/Recall Curve</span></h3>""", unsafe_allow_html=True)

st.divider()

# This is the 6th container of the page that allows the user to write and present their own Python code.
# This container gives additional freedom to the user.
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Additional Analysis</span></h3>""", unsafe_allow_html=True)

st.divider()
