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
from utils.eval_per_query import *

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
            if 'qme_random_state' not in st.session_state:
                st.session_state.qme_random_state = 42
            if 'qme_random_size' not in st.session_state:
                st.session_state.qme_random_size = 100

            st.write(f"""<div style="text-align: center;">  ⚠️ Note: Too many available queries (<span style="color:red;">{len(st.session_state.qme_selected_queries)}</span>).
            To enhance the following analysis, a random set will be used. Please select the following:</div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.session_state.qme_random_size = st.number_input(
                    "Set Number of queries to be randomly selected (default: 49)",
                    min_value=1,
                    value=st.session_state.qme_random_size,
                    max_value=len(st.session_state.qme_selected_queries),
                    step=1
                )
            with col2:
                st.session_state.qme_random_state = st.number_input(
                    "Set random state for query selection (default: 42)",
                    min_value=1,
                    value=st.session_state.qme_random_state,
                    max_value=100,
                    step=1
                )

            # Main content logic
            st.write(f"""<div style="text-align: center;"> A total of <span style="color:red;">{st.session_state.qme_random_size}</span> random queries have been
            selected based on a random state equal to <span style="color:red;">{st.session_state.qme_random_state}</span> and will be used for the upcoming analyses.</div>""", unsafe_allow_html=True)

            st.session_state.qme_selected_queries_random = st.session_state.qme_selected_queries.sample(n=st.session_state.qme_random_size, random_state=st.session_state.qme_random_state)

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
    _, _, custom_user, default_measures, _, _ = return_available_measures()

    if 'qme_selected_qrels' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")
    else:
        # Get the list of selected run files
        selected_runs_files = list(st.session_state.qme_selected_runs.keys())

        if st.session_state.qme_max_relevance >= 2:
            st.session_state.qme_relevance_threshold = st.slider(
                "Select from the Available Relevance Thresholds (Slide)",
                min_value=1,
                max_value=2,
                value=1,
                key="me_slider3",
                help='Fragment rerun'
            )

            if 'qme_prev_relevance_threshold' not in st.session_state:
                st.session_state.qme_prev_relevance_threshold = 1

            if st.session_state.qme_relevance_threshold != st.session_state.qme_prev_relevance_threshold:
                st.session_state.qme_prev_relevance_threshold = st.session_state.qme_relevance_threshold
        else:
            st.session_state.qme_relevance_threshold = 1
            st.write("""**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""", unsafe_allow_html=True)

            # Create columns
        col1, col2 = st.columns(2)  # Adjust the column width ratio as needed

        with col1:

            # Initialize session state variables if they don't exist
            if 'qme_selected_measures' not in st.session_state:
                st.session_state.qme_selected_measures = custom_user[0:4]  # Default selected measures

            selected_measures = st.multiselect("Select additional measures:", custom_user, default=custom_user[1:5], help='Fragment rerun')

        with col2:
            if 'qme_selected_cutoff' not in st.session_state:
                st.session_state.qme_selected_cutoff = 10  # Default cutoff value

            selected_cutoff = st.number_input("Enter cutoff value:", min_value=1, value=10, max_value=1000, step=1, help='Fragment rerun')

            # Update session state with current selections
            st.session_state.qme_selected_measures = selected_measures
            st.session_state.qme_selected_cutoff = selected_cutoff

        if len(st.session_state.qme_selected_measures) >= 1:
            results = per_query_evaluation(st.session_state.qme_selected_qrels, st.session_state.qme_selected_runs, st.session_state.qme_selected_measures, st.session_state.qme_relevance_threshold,
                                           st.session_state.qme_selected_cutoff,
                                           None, None)

            if len(st.session_state.qme_selected_runs) > 1:
                # Analyze the results
                analysis_results = analyze_performance_perq(results)

                # Display the analysis results in two columns
                st.header("Result Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                    This analysis identifies two aspects of the retrieval performance:
                    1. Queries with consistent performance across all experiments
                    2. Queries with large performance gaps between experiments
                    """)

                with col2:
                    st.markdown("""
                    For identifying large performance gaps, we use the Interquartile Range (IQR) method:
                    - We calculate the IQR for each measure across all queries and experiments.
                    - A performance gap is considered 'large' if it's greater than 1.5 times the IQR.
                    - This method adapts to each measure's scale and distribution.

                    Large gaps may indicate sensitivity to specific experimental conditions and should be investigated further.

                    [More information about IQR](https://en.wikipedia.org/wiki/Interquartile_range)
                    """)

                # Create a list of measures
                measures = list(analysis_results.keys())

                # Calculate the number of rows needed (ceiling division by 2)
                num_rows = math.ceil(len(measures) / 2)

                # Create rows and columns
                for i in range(num_rows):
                    col1, col2 = st.columns(2)

                    # First column
                    if i * 2 < len(measures):
                        measure = measures[i * 2]
                        with col1:
                            st.subheader(f"Result Analysis based on {measure}")
                            with st.expander("See Analysis"):

                                same_performance = analysis_results[measure]["same_performance"]
                                if same_performance:
                                    st.write(f"**Queries with equal performance across experiments:** {', '.join(map(str, same_performance))}")
                                    st.write("**Possible indications:**")
                                    st.write("- Queries with similar topics")
                                    st.write("- Potential ceiling or floor effects. Indication of easy or hard queries w.r.t. the collection.")
                                else:
                                    st.write("**No queries with consistent performance.**")
                                    st.write(" - Suggests significant variability across experiments.")

                                large_gaps = analysis_results[measure]["large_gaps"]
                                threshold = analysis_results[measure]["threshold"]
                                if large_gaps:
                                    st.write(f"""**Queries with large performance gaps (greater than 1.5xIQR = <span style="color:red;">{threshold:.3f})**</span>:""", unsafe_allow_html=True)
                                    for query_id, min_val, max_val, _ in large_gaps:
                                        st.write(f"  Query {query_id}: min = {min_val:.3f}, max = {max_val:.3f}, gap = {max_val - min_val:.3f}")
                                    st.write("""<span style="color:red;">Possible indications:</span>""", unsafe_allow_html=True)
                                    st.write("- Sensitivity to experimental conditions")
                                    st.write("- Areas for focused improvement, e.g. combination of the evaluated retrieval systems.")
                                else:
                                    st.write("**No queries with large performance gaps.**")
                                    st.write("- Suggests consistent performance across experiments.")

                    # Second column
                    if i * 2 + 1 < len(measures):
                        measure = measures[i * 2 + 1]
                        with col2:
                            st.subheader(f"Result Analysis based on {measure}")
                            with st.expander("See Analysis"):
                                same_performance = analysis_results[measure]["same_performance"]
                                if same_performance:
                                    st.write(f"**Queries with equal performance across experiments:** {', '.join(map(str, same_performance))}")
                                    st.write("""<span style="color:red;">Possible indications:</span>""", unsafe_allow_html=True)
                                    st.write("- Queries with similar topics")
                                    st.write("- Potential ceiling or floor effects. Indication of easy or hard queries w.r.t. the collection.")
                                else:
                                    st.write("**No queries with consistent performance.**")
                                    st.write(" - Suggests significant variability across experiments.")

                                large_gaps = analysis_results[measure]["large_gaps"]
                                threshold = analysis_results[measure]["threshold"]
                                if large_gaps:
                                    st.write(f"""**Queries with large performance gaps (greater than 1.5xIQR = <span style="color:red;">{threshold:.3f})**</span>:""", unsafe_allow_html=True)
                                    for query_id, min_val, max_val, _ in large_gaps:
                                        st.write(f"  Query {query_id}: min = {min_val:.3f}, max = {max_val:.3f}, gap = {max_val - min_val:.3f}")
                                    st.write("""<span style="color:red;">Possible indications:</span>""", unsafe_allow_html=True)
                                    st.write("- Sensitivity to experimental conditions")
                                    st.write("- Areas for focused improvement, e.g. combination of the evaluated retrieval systems.")
                                else:
                                    st.write("**No queries with large performance gaps.**")
                                    st.write(" - Suggests consistent performance across experiments.")
            else:
                st.divider()

        else:
            st.warning("Please select at least one measure to begin your evaluation.", icon="⚠")
            st.divider()

st.divider()

# Per query Measure Performance Plots Comparison with a Baseline Run
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Query-based Experimental Evaluation</span> Vs <span style="color:red;">Baseline</span></h3>""", unsafe_allow_html=True)
    _, _, custom_user, default_measures, _, _ = return_available_measures()

    if 'qme_selected_runs' not in st.session_state or len(st.session_state.qme_selected_runs) < 2:
        st.warning("This analysis requires at least two retrieval experiments to be selected.", icon="⚠")

    else:
        # Get the list of selected run files
        selected_runs_files = list(st.session_state.qme_selected_runs.keys())

        if st.session_state.qme_max_relevance >= 2:
            st.session_state.qme_relevance_threshold = st.slider(
                "Select from the Available Relevance Thresholds (Slide)",
                min_value=1,
                max_value=2,
                value=1,
                key="me_slider2",
                help='Fragment rerun'
            )

            if 'qme_prev_relevance_threshold' not in st.session_state:
                st.session_state.qme_prev_relevance_threshold = 1

            if st.session_state.qme_relevance_threshold != st.session_state.qme_prev_relevance_threshold:
                st.session_state.qme_prev_relevance_threshold = st.session_state.qme_relevance_threshold
        else:
            st.session_state.qme_relevance_threshold = 1
            st.write("""**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""", unsafe_allow_html=True)

        # Create columns
        col1, col2 = st.columns(2)  # Adjust the column width ratio as needed

        with col1:

            # Initialize session state variables if they don't exist
            if 'qme_selected_measures' not in st.session_state:
                st.session_state.qme_selected_measures = custom_user[1:2]  # Default selected measures

            selected_measures = st.multiselect("Select additional measures:", custom_user, default=custom_user[1:3], key="multiselect_3", help='Fragment rerun')

        with col2:
            if 'qme_selected_cutoff' not in st.session_state:
                st.session_state.qme_selected_cutoff = 10  # Default cutoff value

            selected_cutoff = st.number_input("Enter cutoff value:", min_value=1, value=10, max_value=1000, step=1, key="cutoff_3", help='Fragment rerun')

            # Update session state with current selections
            st.session_state.qme_selected_measures = selected_measures
            st.session_state.qme_selected_cutoff = selected_cutoff

        if selected_runs_files:
            # Add a selectbox to choose the baseline run
            st.session_state.qme_baseline = st.selectbox(
                "Select a baseline run file:",
                list(st.session_state.qme_selected_runs.keys()),
                help='Fragment rerun'
            )

        results = per_query_evaluation(st.session_state.qme_selected_qrels, st.session_state.qme_selected_runs, st.session_state.qme_selected_measures, st.session_state.qme_relevance_threshold,
                                       st.session_state.qme_selected_cutoff,
                                       st.session_state.qme_baseline, None)

        if len(st.session_state.qme_selected_runs) > 1:
            # Perform analysis
            analysis_results, baseline_run = analyze_performance_difference(results)
            # Display summary statistics and analysis
            st.header("Performance Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                This analysis compares each run against the baseline, identifying:
                1. Percentage of queries improved, degraded, or unchanged
                2. Average and median differences in performance
                3. Variability in performance differences across queries

                High improvement percentages may indicate:
                - More effective retrieval methods for these queries
                - Potential areas where the new approach excels
                """)

            with col2:
                st.markdown("""
                The analysis also provides insights on:
                - Overall trend of improvement or degradation
                - Presence of extreme values affecting the results
                - Variability in performance differences

                Large variability or presence of extreme values may suggest:
                - Inconsistent performance across different types of queries
                - Potential areas for focused improvement or investigation
                """)

            # Calculate the number of runs
            num_runs = len(analysis_results)

            # Create a row for each run
            for run, run_analysis in analysis_results.items():
                st.subheader(f"{run} vs {baseline_run}")

                # Calculate the number of measures
                measures = list(run_analysis.keys())
                num_measures = len(measures)
                num_rows = math.ceil(num_measures / 2)

                # Create rows and columns for measures
                for i in range(num_rows):
                    col1, col2 = st.columns(2)

                    # First column
                    if i * 2 < num_measures:
                        measure = measures[i * 2]
                        analysis = run_analysis[measure]
                        with col1:
                            st.subheader(f"Analysis for {measure}")
                            st.write(f"Improved Queries: {len(analysis['improved_queries'])} ({analysis['pct_improved']:.2f}%)")
                            st.write(f"Degraded Queries: {len(analysis['degraded_queries'])} ({analysis['pct_degraded']:.2f}%)")
                            st.write(f"Unchanged Queries: {len(analysis['unchanged_queries'])} ({analysis['pct_unchanged']:.2f}%)")
                            with st.expander("See detailed analysis"):
                                st.write(f"Average Difference: {analysis['avg_diff']:.3f}")
                                st.write(f"Median Difference: {analysis['median_diff']:.3f}")
                                st.write(f"Standard Deviation of Difference: {analysis['std_diff']:.3f}")

                                # Additional insights
                                insights = []
                                if analysis['pct_improved'] > analysis['pct_degraded']:
                                    insights.append(f"This run shows overall improvement over the baseline for {measure}, with {analysis['pct_improved']:.2f}% of queries improved.")
                                elif analysis['pct_improved'] < analysis['pct_degraded']:
                                    insights.append(f"This run shows overall degradation compared to the baseline for {measure}, with {analysis['pct_degraded']:.2f}% of queries degraded.")
                                else:
                                    insights.append(f"This run shows no significant overall difference from the baseline for {measure}.")

                                if analysis['avg_diff'] > 0:
                                    insights.append(f"The average difference is positive ({analysis['avg_diff']:.3f}), indicating an overall improvement.")
                                elif analysis['avg_diff'] < 0:
                                    insights.append(f"The average difference is negative ({analysis['avg_diff']:.3f}), indicating an overall degradation.")

                                if abs(analysis['median_diff']) > abs(analysis['avg_diff']):
                                    insights.append("The median difference is larger than the average, suggesting some extreme values are influencing the results.")

                                if analysis['std_diff'] > abs(analysis['avg_diff']):
                                    insights.append("High variability in differences across queries. Some queries may have significantly larger improvements or degradations than others.")

                                for insight in insights:
                                    st.write(f"- {insight}")

                                st.write("Specific Queries:")
                                st.write(f"- Improved: {', '.join(map(str, analysis['improved_queries']))}")
                                st.write(f"- Degraded: {', '.join(map(str, analysis['degraded_queries']))}")
                                st.write(f"- Unchanged: {', '.join(map(str, analysis['unchanged_queries']))}")

                    # Second column
                    if i * 2 + 1 < num_measures:
                        measure = measures[i * 2 + 1]
                        analysis = run_analysis[measure]
                        with col2:
                            st.subheader(f"Analysis for {measure}")
                            st.write(f"Improved Queries: {len(analysis['improved_queries'])} ({analysis['pct_improved']:.2f}%)")
                            st.write(f"Degraded Queries: {len(analysis['degraded_queries'])} ({analysis['pct_degraded']:.2f}%)")
                            st.write(f"Unchanged Queries: {len(analysis['unchanged_queries'])} ({analysis['pct_unchanged']:.2f}%)")
                            with st.expander("See detailed analysis"):
                                st.write(f"Average Difference: {analysis['avg_diff']:.3f}")
                                st.write(f"Median Difference: {analysis['median_diff']:.3f}")
                                st.write(f"Standard Deviation of Difference: {analysis['std_diff']:.3f}")

                                # Additional insights
                                insights = []
                                if analysis['pct_improved'] > analysis['pct_degraded']:
                                    insights.append(f"This run shows overall improvement over the baseline for {measure}, with {analysis['pct_improved']:.2f}% of queries improved.")
                                elif analysis['pct_improved'] < analysis['pct_degraded']:
                                    insights.append(f"This run shows overall degradation compared to the baseline for {measure}, with {analysis['pct_degraded']:.2f}% of queries degraded.")
                                else:
                                    insights.append(f"This run shows no significant overall difference from the baseline for {measure}.")

                                if analysis['avg_diff'] > 0:
                                    insights.append(f"The average difference is positive ({analysis['avg_diff']:.3f}), indicating an overall improvement.")
                                elif analysis['avg_diff'] < 0:
                                    insights.append(f"The average difference is negative ({analysis['avg_diff']:.3f}), indicating an overall degradation.")

                                if abs(analysis['median_diff']) > abs(analysis['avg_diff']):
                                    insights.append("The median difference is larger than the average, suggesting some extreme values are influencing the results.")

                                if analysis['std_diff'] > abs(analysis['avg_diff']):
                                    insights.append("High variability in differences across queries. Some queries may have significantly larger improvements or degradations than others.")

                                for insight in insights:
                                    st.write(f"- {insight}")

                                st.write("Specific Queries:")
                                st.write(f"- Improved: {', '.join(map(str, analysis['improved_queries']))}")
                                st.write(f"- Degraded: {', '.join(map(str, analysis['degraded_queries']))}")
                                st.write(f"- Unchanged: {', '.join(map(str, analysis['unchanged_queries']))}")

st.divider()

