import os

import pandas as pd
import streamlit as st
from utils.data_handler import load_qrel_data, load_query_data, load_run_data
from utils.eval_core import (evaluate_single_run, evaluate_single_run_custom,
                             generate_prec_recall_graphs,
                             get_relevant_and_unjudged,
                             return_available_measures)
from utils.plots import (plot_dist_of_retrieved_docs,
                         plot_precision_recall_curve)
from utils.ui import load_css

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Load custom CSS
load_css("aspire/css/styles.css")

# In this page, the user can load a single experiment and create an Evaluation Report for this experiment. Usage:
# This report can be useful if the user wants to discuss the results with others by sending them a single PDF file
st.markdown(
    """<div style="text-align: center;"><h1>Retrieval Evaluation Report<h1></div>""",
    unsafe_allow_html=True,
)

# This is the 1st container of the page, responsible for loading the data that will be used in the Analysis and reporting.
# It leverages the data loading functions implemented in the utils.data folder. By default, the system will display all
# available files. In case a file is missing appropriate messages are shown and the system stops.
with st.container():
    st.markdown(
        "<h3>Loading Queries, Qrels, Runs for Analysis</h3>", unsafe_allow_html=True
    )
    columns = st.columns(3)

    # File selections for each folder
    with columns[0]:
        queries_dir = "retrieval_experiments/queries"
        queries_file = st.selectbox(
            "Select Queries",
            os.listdir(queries_dir) if os.path.exists(queries_dir) else [],
        )

    with columns[1]:
        qrels_dir = "retrieval_experiments/qrels"
        qrels_file = st.selectbox(
            "Select Qrels", os.listdir(qrels_dir) if os.path.exists(qrels_dir) else []
        )

    with columns[2]:
        runs_dir = "retrieval_experiments/retrieval_runs"
        st.session_state.runs_file = st.selectbox(
            "Select Retrieval Run",
            os.listdir(runs_dir) if os.path.exists(runs_dir) else [],
        )

# Button to start evaluation
if st.button("Begin the Experimental Evaluation!", key="stButtonCenter"):
    if qrels_file:
        st.session_state.selected_qrels = load_qrel_data(
            os.path.join(qrels_dir, qrels_file)
        )
        st.session_state.max_relevance = st.session_state.selected_qrels[
            "relevance"
        ].max()
    else:
        st.write("Please select Qrels file to proceed.")
        st.stop()

    if st.session_state.runs_file:
        st.session_state.selected_runs = load_run_data(
            os.path.join(runs_dir, st.session_state.runs_file)
        )
    else:
        st.write("Please select Retrieval Run file to proceed.")
        st.stop()

    st.markdown(
        f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{st.session_state.runs_file.replace('.txt', '').replace('.csv', '')}</span> experiment using the <span style="color:red;">{qrels_file}</span> qrels.</div>""",
        unsafe_allow_html=True,
    )

    if queries_file:
        st.session_state.selected_queries = load_query_data(
            os.path.join(queries_dir, queries_file)
        )
        st.markdown(
            f"""<div style="text-align: center;">This experiment is associated with the <span style="color:red;">{queries_file}</span> queries.</div>""",
            unsafe_allow_html=True,
        )

st.divider()

# Overall Retrieval Characteristics
with st.container():
    st.markdown(
        """<h3>Retrieval Performance - <span style="color:red;">Overall Retrieval Characteristics</span></h3>""",
        unsafe_allow_html=True,
    )

    if "selected_qrels" not in st.session_state:
        st.warning(
            "Please select retrieval experiment and qrels to begin your evaluation.",
            icon="⚠",
        )

    else:
        if st.session_state.max_relevance >= 2:
            st.session_state.relevance_threshold = st.slider(
                "Select from the Available Relevance Thresholds (Slide)",
                min_value=1,
                max_value=2,
                value=1,
                key="me_slider4",
            )

            if "prev_relevance_threshold" not in st.session_state:
                st.session_state.prev_relevance_threshold = 1

            if (
                st.session_state.relevance_threshold
                != st.session_state.prev_relevance_threshold
            ):
                st.session_state.prev_relevance_threshold = (
                    st.session_state.relevance_threshold
                )
        else:
            st.session_state.relevance_threshold = 1
            st.write(
                """**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""",
                unsafe_allow_html=True,
            )

        columns = st.columns(2)
        _, _, _, _, overall_measures, precision_measures = return_available_measures()

        with columns[0]:
            if (
                "selected_qrels" in st.session_state
                and not st.session_state.selected_qrels.empty
            ):
                overall_measures_results = {}
                for measure_name in overall_measures:
                    overall_measures_results[measure_name] = evaluate_single_run(
                        st.session_state.selected_qrels,
                        st.session_state.selected_runs,
                        measure_name,
                        st.session_state.relevance_threshold,
                    )

                df_measures = pd.DataFrame(
                    [overall_measures_results],
                    index=[
                        st.session_state.runs_file.replace(".txt", "").replace(
                            ".csv", ""
                        )
                    ],
                )

                # Rename columns with custom names
                df_measures = df_measures.rename(
                    columns={
                        "NumQ": "Total Queries",
                        # "NumRet": "Retrieved Documents",
                        "NumRel": "Relevant Documents",
                        "NumRelRet": "Relevant Retrieved Documents",
                    }
                )

                st.dataframe(df_measures.transpose(), use_container_width=True)

                recall_measures = {
                    "Recall@50": evaluate_single_run(
                        st.session_state.selected_qrels,
                        st.session_state.selected_runs,
                        "R@50",
                        st.session_state.relevance_threshold,
                    ),
                    "Recall@1000": evaluate_single_run(
                        st.session_state.selected_qrels,
                        st.session_state.selected_runs,
                        "R@1000",
                        st.session_state.relevance_threshold,
                    ),
                }
                df_recall_measures = pd.DataFrame(
                    [recall_measures],
                    index=[
                        st.session_state.runs_file.replace(".txt", "").replace(
                            ".csv", ""
                        )
                    ],
                )
                st.dataframe(df_recall_measures, use_container_width=True)

        with columns[1]:
            if (
                "selected_qrels" in st.session_state
                and not st.session_state.selected_qrels.empty
            ):
                precision_measures_results = {}
                for measure_name in precision_measures:
                    precision_measures_results[measure_name] = evaluate_single_run(
                        st.session_state.selected_qrels,
                        st.session_state.selected_runs,
                        measure_name,
                        st.session_state.relevance_threshold,
                    )

                df_prec_measures = pd.DataFrame(
                    [precision_measures_results],
                    index=[
                        st.session_state.runs_file.replace(".txt", "").replace(
                            ".csv", ""
                        )
                    ],
                ).transpose()

                st.dataframe(df_prec_measures, use_container_width=True)

st.divider()

# Common Evaluation Measures
with st.container():
    st.markdown(
        """<h3>Retrieval Performance - <span style="color:red;">Experimental Evaluation</span></h3>""",
        unsafe_allow_html=True,
    )
    _, _, custom_user, default_measures, _, _ = return_available_measures()

    if "selected_qrels" not in st.session_state:
        st.warning(
            "Please select retrieval experiment and qrels to begin your evaluation.",
            icon="⚠",
        )

    else:
        if st.session_state.max_relevance >= 2:
            st.session_state.relevance_threshold = st.slider(
                "Select from the Available Relevance Thresholds (Slide)",
                min_value=1,
                max_value=2,
                value=1,
                key="me_slider5",
            )

            if "prev_relevance_threshold" not in st.session_state:
                st.session_state.prev_relevance_threshold = 1

            if (
                st.session_state.relevance_threshold
                != st.session_state.prev_relevance_threshold
            ):
                st.session_state.prev_relevance_threshold = (
                    st.session_state.relevance_threshold
                )
        else:
            st.session_state.relevance_threshold = 1
            st.write(
                """**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""",
                unsafe_allow_html=True,
            )

        if (
            "selected_qrels" in st.session_state
            and not st.session_state.selected_qrels.empty
        ):
            freq_measures_results = {}
            for measure_name in default_measures:
                freq_measures_results[measure_name] = evaluate_single_run(
                    st.session_state.selected_qrels,
                    st.session_state.selected_runs,
                    measure_name,
                    st.session_state.relevance_threshold,
                )

            common_measures = pd.DataFrame(
                [freq_measures_results],
                index=[
                    st.session_state.runs_file.replace(".txt", "").replace(".csv", "")
                ],
            )
            st.dataframe(common_measures, use_container_width=True)

            st.divider()

            # Splitting the container
            col = st.columns(2)

            # Initialize session state variables if they don't exist
            if "selected_measures" not in st.session_state:
                st.session_state.selected_measures = custom_user[
                    0:1
                ]  # Default selected measures
            if "selected_cutoff" not in st.session_state:
                st.session_state.selected_cutoff = 25  # Default cutoff value

            with col[0]:
                selected_measures = st.multiselect(
                    "Select additional measures:", custom_user, default=custom_user[5:6]
                )
            with col[1]:
                selected_cutoff = st.number_input(
                    "Enter cutoff value:", min_value=1, value=25, max_value=1000, step=1
                )

                # Update session state with current selections
            st.session_state.selected_measures = selected_measures
            st.session_state.selected_cutoff = selected_cutoff

            # Evaluate the experiment whenever selections change
            users_eval = {}
            for user_metric in selected_measures:
                user_metric_name, user_metric_score = evaluate_single_run_custom(
                    st.session_state.selected_qrels,
                    st.session_state.selected_runs,
                    user_metric,
                    selected_cutoff,
                    st.session_state.relevance_threshold,
                )
                users_eval[str(user_metric_name)] = user_metric_score

            # Convert the dictionary to a DataFrame
            user_measures = pd.DataFrame(
                [users_eval],
                index=[
                    st.session_state.runs_file.replace(".txt", "").replace(".csv", "")
                ],
            )

            # Display the DataFrame
            st.dataframe(user_measures, use_container_width=True)

            # Store user_measures_eval in session state for further use
            st.session_state.user_measures_eval = user_measures.copy()

st.divider()

# Positional Distribution of Relevant Retrieved Documents
with st.container():
    st.markdown(
        """<h3>Retrieval Performance - <span style="color:red;">Positional Distribution of Relevant and Unjudged Retrieved Documents</span></h3>""",
        unsafe_allow_html=True,
    )

    if "selected_qrels" not in st.session_state:
        st.warning(
            "Please select retrieval experiment and qrels to begin your evaluation.",
            icon="⚠",
        )

    else:
        ranking_per_relevance = get_relevant_and_unjudged(
            st.session_state.selected_qrels, st.session_state.selected_runs
        )

        plot_dist_of_retrieved_docs(ranking_per_relevance)

st.divider()

# Precision/Recall Curve
with st.container():
    st.markdown(
        """<h3>Retrieval Performance - <span style="color:red;">Precision/Recall Curve</span></h3>""",
        unsafe_allow_html=True,
    )

    if "selected_qrels" not in st.session_state:
        st.warning(
            "Please select retrieval experiment and qrels to begin your evaluation.",
            icon="⚠",
        )
    else:
        if st.session_state.max_relevance >= 2:
            st.session_state.relevance_threshold = st.slider(
                "Select from the Available Relevance Thresholds (Slide)",
                min_value=1,
                max_value=2,
                value=1,
                key="me_slider6",
            )

            if "prev_relevance_threshold" not in st.session_state:
                st.session_state.prev_relevance_threshold = 1

            if (
                st.session_state.relevance_threshold
                != st.session_state.prev_relevance_threshold
            ):
                st.session_state.prev_relevance_threshold = (
                    st.session_state.relevance_threshold
                )
        else:
            st.session_state.relevance_threshold = 1
            st.write(
                """**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""",
                unsafe_allow_html=True,
            )

        prec_recall_graphs = generate_prec_recall_graphs(
            st.session_state.relevance_threshold,
            st.session_state.selected_qrels,
            st.session_state.selected_runs,
        )

        plot_precision_recall_curve(
            prec_recall_graphs, st.session_state.relevance_threshold
        )

st.divider()

# Additional Analysis
with st.container():
    st.markdown(
        """<h3>Retrieval Performance - <span style="color:red;">Personal Notes</span></h3>""",
        unsafe_allow_html=True,
    )

    st.text_area(
        "Please add additional comments regarding this experiment.",
        "",
        key="placeholder",
    )
st.divider()

st.markdown(
    """<h5 style="text-align:center;"><span style="color:red;">To export the report as PDF press (⌘+P or Ctrl+P)</span></h5>""",
    unsafe_allow_html=True,
)
