import os
import pandas as pd
import streamlit as st
from utils.data import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.evaluation_measures import evaluate_single_run, return_available_measures

# Load custom CSS
load_css("css/styles.css")

# In this page, the user can load a single experiment and create an Evaluation Report for this experiment. Usage:
# This report can be useful if the user wants to discuss the results with others by sending them a single PDF file
st.markdown("""<div style="text-align: center;"><h1>Retrieval Evaluation Report<h1></div>""", unsafe_allow_html=True)

# This is the 1st container of the page, responsible for loading the data that will be used in the Analysis and reporting.
# It leverages the data loading functions implemented in the utils.data folder. By default, the system will display all
# available files. In case a file is missing appropriate messages are shown and the system stops.
with st.container():
    st.markdown("<h3>Loading Queries, Qrels, Runs for Analysis</h3>", unsafe_allow_html=True)
    columns = st.columns(3)

    # File selections for each folder
    with columns[0]:
        queries_dir = "../retrieval_experiments/queries"
        queries_file = st.selectbox("Select Queries", os.listdir(queries_dir) if os.path.exists(queries_dir) else [])

    with columns[1]:
        qrels_dir = "../retrieval_experiments/qrels"
        qrels_file = st.selectbox("Select Qrels", os.listdir(qrels_dir) if os.path.exists(qrels_dir) else [])

    with columns[2]:
        runs_dir = "../retrieval_experiments/retrieval_runs"
        st.session_state.runs_file = st.selectbox("Select Retrieval Run", os.listdir(runs_dir) if os.path.exists(runs_dir) else [])

# Button to start evaluation
if st.button("Begin the Experimental Evaluation!", key='stButtonCenter'):
    if qrels_file:
        st.session_state.selected_qrels = load_qrel_data(os.path.join(qrels_dir, qrels_file))
        st.session_state.max_relevance = st.session_state['selected_qrels']["relevance"].max()
    else:
        st.write("Please select Qrels file to proceed.")
        st.stop()

    if st.session_state.runs_file:
        st.session_state.selected_runs = load_run_data(os.path.join(runs_dir, st.session_state.runs_file))
    else:
        st.write("Please select Retrieval Run file to proceed.")
        st.stop()

    st.markdown(
        f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{st.session_state.runs_file.replace('.txt', '')}</span> experiment using the <span style="color:red;">{qrels_file}</span> qrels.</div>""",
        unsafe_allow_html=True)

    # st.markdown(
    #     f"""<div style="text-align: center;">Relevance Threshold is <span style="color:red;">{st.session_state.relevance_threshold}</span>.</div>""",
    #     unsafe_allow_html=True)

    if queries_file:
        st.session_state.selected_queries = load_query_data(os.path.join(queries_dir, queries_file))
        st.markdown(
            f"""<div style="text-align: center;">This experiment is associated with the <span style="color:red;">{queries_file}</span> queries.</div>""",
            unsafe_allow_html=True,
        )

st.divider()

# Overall Retrieval Characteristics
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Overall Retrieval Characteristics</span></h3>""", unsafe_allow_html=True)

    if 'selected_qrels' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")

    else:
        st.session_state.relevance_threshold = st.slider(
            "Select from the Available Relevance Thresholds (Slide)",
            min_value=1,
            max_value=2,
            value=1,
            key="slider1"
        )

        if 'prev_relevance_threshold' not in st.session_state:
            st.session_state.prev_relevance_threshold = 1

        if st.session_state.relevance_threshold != st.session_state.prev_relevance_threshold:
            st.session_state.prev_relevance_threshold = st.session_state.relevance_threshold

        columns = st.columns(2)
        _, _, _, _, overall_measures, precision_measures = return_available_measures()

        with columns[0]:
            if 'selected_qrels' in st.session_state and not st.session_state.selected_qrels.empty:
                overall_measures_results = {}
                for measure_name in overall_measures:
                    overall_measures_results[measure_name] = evaluate_single_run(
                        st.session_state.selected_qrels, st.session_state.selected_runs, measure_name, st.session_state.relevance_threshold
                    )

                df_measures = pd.DataFrame([overall_measures_results],
                                           index=[st.session_state.runs_file.replace('.txt', '')])

                # Rename columns with custom names
                # TODO: FIX! There is an error in the calculation of the retrieved documents. It should be 1000x the query number but it is equal to the numrelret.
                df_measures = df_measures.rename(columns={
                    "NumQ": "Total Queries",
                    "NumRet": "Retrieved Documents",
                    "NumRel": "Relevant Documents",
                    "NumRelRet": "Relevant Retrieved Documents"
                })

                st.dataframe(df_measures.transpose(), use_container_width=True)

        with columns[1]:
            if 'selected_qrels' in st.session_state and not st.session_state.selected_qrels.empty:
                precision_measures_results = {}
                for measure_name in precision_measures:
                    precision_measures_results[measure_name] = evaluate_single_run(
                        st.session_state.selected_qrels, st.session_state.selected_runs, measure_name, st.session_state.relevance_threshold
                    )
                #TODO: Add a caption on the table to say that the values are across queries or rename them.
                df_prec_measures = pd.DataFrame([precision_measures_results],
                                                index=[st.session_state.runs_file.replace('.txt', '')]).transpose()
                st.dataframe(df_prec_measures, use_container_width=True)

st.divider()

# Common Evaluation Measures
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Common Evaluation Measures</span></h3>""", unsafe_allow_html=True)
    _, _, _, default_measures, _, _ = return_available_measures()

    if 'selected_qrels' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")

    else:
        st.session_state.relevance_threshold = st.slider(
            "Select from the Available Relevance Thresholds (Slide)",
            min_value=1,
            max_value=2,
            value=1,
            key="slider2"
        )

        if 'prev_relevance_threshold' not in st.session_state:
            st.session_state.prev_relevance_threshold = 1

        if st.session_state.relevance_threshold != st.session_state.prev_relevance_threshold:
            st.session_state.prev_relevance_threshold = st.session_state.relevance_threshold

        if 'selected_qrels' in st.session_state and not st.session_state.selected_qrels.empty:
            freq_measures_results = {}
            for measure_name in default_measures:
                freq_measures_results[measure_name] = evaluate_single_run(
                    st.session_state.selected_qrels, st.session_state.selected_runs, measure_name, st.session_state.relevance_threshold
                )
            common_measures = pd.DataFrame([freq_measures_results], index=[st.session_state.runs_file.replace('.txt', '')])
            st.dataframe(common_measures, use_container_width=True, )

st.divider()

# Positional Distribution of Relevant Retrieved Documents
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Positional Distribution of Relevant Retrieved Documents</span></h3>""", unsafe_allow_html=True)

st.divider()

# Precision/Recall Curve
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Precision/Recall Curve</span></h3>""", unsafe_allow_html=True)

st.divider()

# Additional Analysis
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Additional Analysis</span></h3>""", unsafe_allow_html=True)

st.divider()
