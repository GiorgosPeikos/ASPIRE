import os
import pandas as pd
import streamlit as st
from utils.data import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.evaluation_measures import evaluate_single_run, return_available_measures, evaluate_multiple_runs
from utils.plots import plot_precision_recall_curve

# Set the page configuration to wide mode
# st.set_page_config(layout="wide")

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
    st.session_state.me_selected_qrels = load_qrel_data(os.path.join(qrels_dir, qrels_file))
    st.session_state.me_max_relevance = st.session_state.me_selected_qrels["relevance"].max()

    # Load selected runs data
    st.session_state.me_selected_runs = {}
    for run_file in selected_runs_files:
        st.session_state.me_selected_runs[run_file] = load_run_data(os.path.join(runs_dir, run_file))

    if st.session_state.me_selected_runs:
        st.markdown(
            f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{", ".join(selected_runs_files).replace('.txt', '')}</span> experiments using the <span style="color:red;">{qrels_file}</span> qrels.</div>""",
            unsafe_allow_html=True)

    if queries_file:
        st.session_state.me_selected_queries = load_query_data(os.path.join(queries_dir, queries_file))
        st.markdown(
            f"""<div style="text-align: center;">This experiment is associated with the <span style="color:red;">{queries_file}</span> queries.</div>""",
            unsafe_allow_html=True)

st.divider()


# Overall Retrieval Characteristics
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Overall Retrieval Characteristics</span></h3>""", unsafe_allow_html=True)

    if 'me_selected_qrels' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")
    else:
        st.session_state.me_relevance_threshold = st.slider(
            "Select from the Available Relevance Thresholds (Slide)",
            min_value=1,
            max_value=2,
            value=1,
            key="me_slider1"
        )

        if 'me_prev_relevance_threshold' not in st.session_state:
            st.session_state.me_prev_relevance_threshold = 1

        if st.session_state.me_relevance_threshold != st.session_state.me_prev_relevance_threshold:
            st.session_state.me_prev_relevance_threshold = st.session_state.me_relevance_threshold

        # Get the list of selected run files
        me_selected_runs_files = list(st.session_state.me_selected_runs.keys())

        if me_selected_runs_files:
            columns = st.columns(2)
            _, _, _, _, overall_measures, precision_measures = return_available_measures()

            overall_measures_combined = pd.DataFrame()
            precision_measures_combined = pd.DataFrame()
            recall_measures_combined = pd.DataFrame()  # New DataFrame for recall measures

            progress_bar = st.empty()
            progress_text = st.empty()
            total_files = len(selected_runs_files)

            for i, run_file in enumerate(selected_runs_files):
                progress = (i + 1) / total_files
                progress_bar.progress(progress)
                progress_text.text(f"Evaluating experiment {i + 1}/{total_files}")

                with columns[0]:
                    # Removed experiment name display

                    overall_measures_results = {}
                    for measure_name in overall_measures:
                        overall_measures_results[measure_name] = evaluate_single_run(
                            st.session_state.me_selected_qrels, st.session_state.me_selected_runs[run_file], measure_name, st.session_state.me_relevance_threshold
                        )

                    df_measures = pd.DataFrame([overall_measures_results],
                                               index=[run_file.replace('.txt', '')])

                    df_measures = df_measures.rename(columns={
                        "NumQ": "Total Queries",
                        "NumRet": "Retrieved Documents",
                        "NumRel": "Relevant Documents",
                        "NumRelRet": "Relevant Retrieved Documents"
                    })

                    # Add overall measures to the combined DataFrame
                    overall_measures_combined = pd.concat([overall_measures_combined, df_measures.transpose()], axis=1)

                with columns[1]:
                    precision_measures_results = {}
                    for measure_name in precision_measures:
                        precision_measures_results[measure_name] = evaluate_single_run(
                            st.session_state.me_selected_qrels, st.session_state.me_selected_runs[run_file], measure_name, st.session_state.me_relevance_threshold
                        )

                    df_prec_measures = pd.DataFrame(precision_measures_results, index=[run_file.replace('.txt', '')]).transpose()

                    # Add precision measures to the combined DataFrame
                    precision_measures_combined[run_file.replace('.txt', '')] = df_prec_measures.iloc[:, 0]

                    # Calculate and add recall measures to the combined DataFrame
                    recall_measures_results = {
                        "Recall@50": evaluate_single_run(
                            st.session_state.me_selected_qrels, st.session_state.me_selected_runs[run_file], "R@50", st.session_state.me_relevance_threshold
                        ),
                        "Recall@1000": evaluate_single_run(
                            st.session_state.me_selected_qrels, st.session_state.me_selected_runs[run_file], "R@1000", st.session_state.me_relevance_threshold
                        )
                    }
                    df_recall_measures = pd.DataFrame([recall_measures_results], index=[run_file.replace('.txt', '')])
                    recall_measures_combined = pd.concat([recall_measures_combined, df_recall_measures.transpose()], axis=1)

            # Clear the progress bar and text
            progress_bar.empty()
            progress_text.empty()

            # Display the combined DataFrames
            st.markdown("<h4>Overall Measures Combined</h4>", unsafe_allow_html=True)
            st.dataframe(overall_measures_combined, use_container_width=True)

            st.markdown("<h4>Precision Measures Combined</h4>", unsafe_allow_html=True)
            st.dataframe(precision_measures_combined, use_container_width=True)

            st.markdown("<h4>Recall Measures Combined</h4>", unsafe_allow_html=True)
            st.dataframe(recall_measures_combined, use_container_width=True)

st.divider()

# Common Evaluation Measures
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Experimental Evaluation</span></h3>""", unsafe_allow_html=True)
    _, _, custom_user, default_measures, _, _ = return_available_measures()

    if 'me_selected_qrels' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")
    else:
        st.session_state.me_relevance_threshold = st.slider(
            "Select from the Available Relevance Thresholds (Slide)",
            min_value=1,
            max_value=2,
            value=1,
            key="me_slider2"
        )

        if 'me_prev_relevance_threshold' not in st.session_state:
            st.session_state.me_prev_relevance_threshold = 1

        if st.session_state.me_relevance_threshold != st.session_state.me_prev_relevance_threshold:
            st.session_state.me_prev_relevance_threshold = st.session_state.me_relevance_threshold

        # Get the list of selected run files
        selected_runs_files = list(st.session_state.me_selected_runs.keys())

        if 'me_selected_qrels' in st.session_state and not st.session_state.me_selected_qrels.empty:
            freq_measures_results = evaluate_multiple_runs(st.session_state.me_selected_qrels, st.session_state.me_selected_runs, default_measures, st.session_state.me_relevance_threshold )

        st.write(freq_measures_results)


st.divider()

