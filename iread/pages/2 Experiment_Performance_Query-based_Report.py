import math
import os
import numpy as np
import streamlit as st
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed")
from utils.data_handler import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.eval_core import return_available_measures
from utils.eval_per_query import per_query_evaluation, analyze_performance_difference_median, analyze_performance_difference, analyze_performance_difference_threshold, analyze_performance_perq
from utils.plots import plot_query_relevance_judgements
from utils.eval_query_collection import analyze_query_judgements

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
        if 'qme_selected_queries_random' not in st.session_state:
            st.session_state.qme_selected_queries_random = st.session_state.qme_selected_queries
            st.write(f"""<div style="text-align: center;"> All <span style="color:red;">{len(st.session_state.qme_selected_queries_random)}</span> provided queries will be used for the 
            following analyses.</div>""", unsafe_allow_html=True)

st.divider()

# Per query Relevance Judgements
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Relevance Judgments per Query</span></h3>""", unsafe_allow_html=True)

    if 'qme_selected_runs' not in st.session_state:
        st.warning("Please select a set of queries to begin your evaluation.", icon="⚠")
    else:
        results = plot_query_relevance_judgements(st.session_state.qme_selected_qrels)
        analysis_results = analyze_query_judgements(results)

        # Display the analysis results
        st.write("""<h4><span style="color:red;">Query Judgement Analysis Results</span></h4>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                        This analysis provides insights into the query judgements:
                        1. Distribution of relevance judgements across queries
                        2. Identification of queries with extreme judgement counts
                        3. Classification of queries as hard or easy
                        """)

        with col2:
            st.markdown("""
                        4. Overall statistics on relevance judgements

                        These insights can help in understanding the characteristics of your query set and potential biases in judgements.
                        """)

        # Create a list of analysis sections
        sections = [
            ("Queries sorted based on Rel. Judgements", 'sorted_queries'),
            ("Relevance Labels - Statistics", 'relevance_stats'),
            ("Query Difficulty", 'query_difficulty'),
            ("Overall Statistics", 'overall_stats')
        ]

        # Calculate the number of rows needed (ceiling division by 2)
        num_rows = math.ceil(len(sections) / 2)

        # Create rows and columns
        for i in range(num_rows):
            col1, col2 = st.columns(2)

            # First column
            if i * 2 < len(sections):
                section_title, section_key = sections[i * 2]
                with col1:
                    st.subheader(section_title)
                    with st.expander("See Analysis"):
                        if section_title == "Queries sorted based on Rel. Judgements":
                            st.markdown("""
                            The queries are sorted based on the number of document judgements for each relevance label, in descending order. 
                            This means that for each label the first query listed has the highest number of judgements for that label.
                            
                            The number in parentheses after each query ID represents the count of judgements for that query.
                            Only the counts for the first two queries are shown to give you an idea of the range.
                            """)

                            for label in analysis_results['sorted_queries']:
                                color = 'red' if label == 'Relevance_Label_0' else 'blue'
                                label_name = 'Irrelevant (Relevance_Label_0)' if label == 'Relevance_Label_0' else label
                                sorted_queries = analysis_results['sorted_queries'][label]

                                # Get the first two queries with their judgement counts
                                top_queries = sorted_queries[:2]
                                top_queries_str = ', '.join([f"{q} (**{count}**)" for q, count in top_queries])

                                # Get the last two queries with their judgement counts
                                bottom_queries = sorted_queries[-2:]
                                bottom_queries_str = ', '.join([f"{q} (**{count}**)" for q, count in bottom_queries])

                                # Get the middle queries without counts
                                middle_queries = [str(q) for q, _ in sorted_queries[2:-2]]

                                # Combine all queries
                                if len(sorted_queries) > 4:
                                    all_queries_str = f"{top_queries_str}, {', '.join(middle_queries)}, {bottom_queries_str}"
                                else:
                                    # If there are 4 or fewer queries, just show all with counts
                                    all_queries_str = ', '.join([f"{q} (**{count}**)" for q, count in sorted_queries])

                                st.markdown(f"<span style='color:{color};'>{label_name}</span>: {all_queries_str}", unsafe_allow_html=True)

                        elif section_title == "Query Difficulty":
                            st.write("""
                            Query difficulty is analyzed based on the number of judgements for each relevance label and the ratio of irrelevant to relevant judgements.
                            For each category:
                            - Hard queries: 5 sample queries with the least judgements per label
                            - Easy queries: 5 sample queries with the most judgements per label
                            """)

                            for label in analysis_results['query_difficulty']:
                                color = 'red' if label == 'Relevance_Label_0' else 'blue'
                                label_name = 'Irrelevant' if label == 'Relevance_Label_0' else label
                                st.markdown(f"""<center><h5><span style='color:{color};'>{label_name}</span></h5></center>""", unsafe_allow_html=True)

                                difficulty = analysis_results['query_difficulty'][label]

                                col1in, col2in = st.columns([1,2])

                                with col1in:
                                    if label == 'Relevance_Label_0':
                                        # Need to inverse the presentation, because queries with many irrelevant are hard
                                        st.write("**Hard queries:**")
                                        st.write(", ".join(map(str, difficulty['easy'])))

                                        st.write("**Easy queries:**")
                                        st.write(", ".join(map(str, difficulty['hard'])))
                                    else:
                                        st.write("**Hard queries:**")
                                        st.write(", ".join(map(str, difficulty['hard'])))

                                        st.write("**Easy queries:**")
                                        st.write(", ".join(map(str, difficulty['easy'])))

                                with col2in:
                                    st.write(f"**Min number of <span style='color:{color};'>{label_name}</span> Judged Documents:** {difficulty['min']:.2f}", unsafe_allow_html=True)
                                    st.write(f"**Max number of  <span style='color:{color};'>{label_name}</span> Judged Documents:** {difficulty['max']:.2f}", unsafe_allow_html=True)

            # Second column
            if i * 2 + 1 < len(sections):
                section_title, section_key = sections[i * 2 + 1]
                with col2:
                    st.subheader(section_title)
                    with st.expander("See Analysis"):
                        if section_title == "Relevance Labels - Statistics":
                            for label in analysis_results:
                                if label.endswith('_stats') and not label.startswith('overall'):
                                    stats = analysis_results[label]
                                    color = 'red' if label == 'Relevance_Label_0_stats' else 'blue'
                                    label_name = 'Irrelevant' if label == 'Relevance_Label_0_stats' else label.split('_stats')[0]
                                    stat_lines = []
                                    for stat in ['mean', 'median', 'std', 'min', 'max']:
                                        if stat in stats:
                                            value = stats[stat]
                                            formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                                            stat_lines.append(f"{stat.capitalize()}: <span style='color:red;'>{formatted_value}</span>")
                                    stats_display = ", ".join(stat_lines)
                                    st.markdown(f"<span style='color:{color};'>{label_name}</span>: {stats_display}", unsafe_allow_html=True)

                        elif section_title == "Overall Statistics":
                            total_judgements = sum(analysis_results['overall_stats'][label] for label in analysis_results['overall_stats'] if not label.endswith('_percentage'))
                            for label in analysis_results['overall_stats']:
                                if not label.endswith('_percentage'):
                                    color = 'red' if label == 'Relevance_Label_0' else 'blue'
                                    label_name = 'Irrelevant' if label == 'Relevance_Label_0' else label
                                    count = analysis_results['overall_stats'][label]
                                    percentage = analysis_results['overall_stats'][f'{label}_percentage']
                                    st.markdown(f"<span style='color:{color}'>{label_name}</span>: **{count}** ({percentage:.2f}% of total)", unsafe_allow_html=True)
                            st.write(f"Total judgements:  **{total_judgements}**", unsafe_allow_html=True)

        st.markdown("### Manually Examine Sampled Queries")
        with st.expander("See Queries"):
            st.dataframe(st.session_state.qme_selected_queries_random[['query_id', 'query_text']], use_container_width=True, hide_index=True)

st.divider()

# Per query Measure Performance Plots
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Experimental Evaluation per Query</span></h3>""", unsafe_allow_html=True)
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

            selected_measures = st.multiselect("Select additional measures:", custom_user, default=custom_user[1:3])

        with col2:
            if 'qme_selected_cutoff' not in st.session_state:
                st.session_state.qme_selected_cutoff = 10  # Default cutoff value

            selected_cutoff = st.number_input("Enter cutoff value:", min_value=1, value=10, max_value=1000, step=1)

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
                st.write("""<h4><span style="color:red;">Information Related to the Analysis</span></h4>""", unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                    This analysis identifies two aspects of the retrieval performance:
                    1. Queries with consistent performance across all experiments
                    2. Queries with large performance gaps between experiments
                    
                    For identifying large performance gaps, we use the Interquartile Range (IQR) method:
                    - We calculate the IQR for each measure across all queries and experiments.
                    - A performance gap is considered 'large' if it's greater than 1.5 times the IQR.
                    """)

                with col2:
                    st.markdown("""
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
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Experimental Evaluation</span> Vs <span style="color:red;">Baseline</span></h3>""", unsafe_allow_html=True)
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

            selected_measures = st.multiselect("Select additional measures:", custom_user, default=custom_user[2:4], key="multiselect_3")

        with col2:
            if 'qme_selected_cutoff' not in st.session_state:
                st.session_state.qme_selected_cutoff = 10  # Default cutoff value

            selected_cutoff = st.number_input("Enter cutoff value:", min_value=1, value=10, max_value=1000, step=1, key="cutoff_3")

            # Update session state with current selections
            st.session_state.qme_selected_measures = selected_measures
            st.session_state.qme_selected_cutoff = selected_cutoff

        if selected_runs_files:
            # Add a selectbox to choose the baseline run
            st.session_state.qme_baseline = st.selectbox(
                "Select a baseline run file:",
                list(st.session_state.qme_selected_runs.keys())
            )

        results = per_query_evaluation(st.session_state.qme_selected_qrels, st.session_state.qme_selected_runs, st.session_state.qme_selected_measures, st.session_state.qme_relevance_threshold,
                                       st.session_state.qme_selected_cutoff,
                                       st.session_state.qme_baseline, None)

        if len(st.session_state.qme_selected_runs) > 1:
            # Perform analysis
            analysis_results, baseline_run = analyze_performance_difference(results)
            # Display summary statistics and analysis
            st.write("""<h4><span style="color:red;">Information Related to the Analysis</span></h4>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                This analysis compares each run against the selected baseline, identifying:
                1. Percentage of queries improved, degraded, or unchanged
                2. Average and median differences in performance
                3. Variability in performance differences across queries
                """)

            with col2:
                st.markdown("""
                The analysis also provides insights on:
                1. Overall trend of improvement or degradation
                2. Presence of extreme values affecting the results
                3. Variability in performance differences
                """)

            # Calculate the number of runs
            num_runs = len(analysis_results)

            # Create a row for each run
            for run, run_analysis in analysis_results.items():
                st.write(f"""<center><h4>{run} <span style="color:red;">vs</span> {baseline_run}</h4></center>""", unsafe_allow_html=True)

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
                            st.subheader(f"Result Analysis based on {measure}")
                            st.write(f"Improved Queries: {len(analysis['improved_queries'])} ({analysis['pct_improved']:.2f}%)")
                            st.write(f"Degraded Queries: {len(analysis['degraded_queries'])} ({analysis['pct_degraded']:.2f}%)")
                            st.write(f"Unchanged Queries: {len(analysis['unchanged_queries'])} ({analysis['pct_unchanged']:.2f}%)")
                            with st.expander("See detailed analysis"):
                                st.write(f"Average Difference: {analysis['avg_diff']:.3f}")
                                st.write(f"Median Difference: {analysis['median_diff']:.3f}")
                                st.write(f"Standard Deviation of Difference: {analysis['std_diff']:.3f}")
                                st.write('----')
                                # Additional insights
                                insights = []
                                if analysis['pct_improved'] > analysis['pct_degraded']:
                                    insights.append(
                                        f"""This run shows overall improvement over the baseline for <span style="color:red;">{measure}</span>, with <span style="color:red;">{analysis['pct_improved']:.2f}</span>% of queries improved.""")
                                elif analysis['pct_improved'] < analysis['pct_degraded']:
                                    insights.append(
                                        f"""This run shows overall degradation compared to the baseline for <span style="color:red;">{measure}</span>, with <span style="color:red;">{analysis['pct_degraded']:.2f}</span>% of queries degraded.""")
                                else:
                                    insights.append(f"""This run shows no significant overall difference from the baseline for <span style="color:red;">{measure}</span>.""")

                                if analysis['avg_diff'] > 0:
                                    insights.append(f"""The average difference is positive (<span style="color:red;">{analysis['avg_diff']:.3f}</span>), indicating an overall improvement.""")
                                elif analysis['avg_diff'] < 0:
                                    insights.append(f"""The average difference is negative (<span style="color:red;">{analysis['avg_diff']:.3f}</span>), indicating an overall degradation.""")

                                if abs(analysis['median_diff']) > abs(analysis['avg_diff']):
                                    insights.append("The median difference is larger than the average, suggesting some extreme values are influencing the results.""")

                                if analysis['std_diff'] > abs(analysis['avg_diff']):
                                    insights.append("""High variability in differences across queries. Some queries may have significantly larger improvements or degradations than others.""")

                                for insight in insights:
                                    st.write(f"- {insight}", unsafe_allow_html=True)

                                st.write("Specifically:")
                                st.write(f"- Improved Queries: {', '.join(map(str, analysis['improved_queries']))}")
                                st.write(f"- Degraded Queries: {', '.join(map(str, analysis['degraded_queries']))}")
                                st.write(f"- Unchanged Queries: {', '.join(map(str, analysis['unchanged_queries']))}")

                    # Second column
                    if i * 2 + 1 < num_measures:
                        measure = measures[i * 2 + 1]
                        analysis = run_analysis[measure]
                        with col2:
                            st.subheader(f"Result Analysis based on {measure}")
                            st.write(f"Improved Queries: {len(analysis['improved_queries'])} ({analysis['pct_improved']:.2f}%)")
                            st.write(f"Degraded Queries: {len(analysis['degraded_queries'])} ({analysis['pct_degraded']:.2f}%)")
                            st.write(f"Unchanged Queries: {len(analysis['unchanged_queries'])} ({analysis['pct_unchanged']:.2f}%)")
                            with st.expander("See detailed analysis"):
                                st.write(f"Average Difference: {analysis['avg_diff']:.3f}")
                                st.write(f"Median Difference: {analysis['median_diff']:.3f}")
                                st.write(f"Standard Deviation of Difference: {analysis['std_diff']:.3f}")
                                st.write('----')
                                # Additional insights
                                insights = []
                                if analysis['pct_improved'] > analysis['pct_degraded']:
                                    insights.append(
                                        f"""This run shows overall improvement over the baseline for <span style="color:red;">{measure}</span>, with <span style="color:red;">{analysis['pct_improved']:.2f}</span>% of queries improved.""")
                                elif analysis['pct_improved'] < analysis['pct_degraded']:
                                    insights.append(
                                        f"""This run shows overall degradation compared to the baseline for <span style="color:red;">{measure}</span>, with <span style="color:red;">{analysis['pct_degraded']:.2f}</span>% of queries degraded.""")
                                else:
                                    insights.append(f"""This run shows no significant overall difference from the baseline for <span style="color:red;">{measure}</span>.""")

                                if analysis['avg_diff'] > 0:
                                    insights.append(f"""The average difference is positive (<span style="color:red;">{analysis['avg_diff']:.3f}</span>), indicating an overall improvement.""")
                                elif analysis['avg_diff'] < 0:
                                    insights.append(f"""The average difference is negative (<span style="color:red;">{analysis['avg_diff']:.3f}</span>), indicating an overall degradation.""")

                                if abs(analysis['median_diff']) > abs(analysis['avg_diff']):
                                    insights.append("The median difference is larger than the average, suggesting some extreme values are influencing the results.""")

                                if analysis['std_diff'] > abs(analysis['avg_diff']):
                                    insights.append("""High variability in differences across queries. Some queries may have significantly larger improvements or degradations than others.""")

                                for insight in insights:
                                    st.write(f"- {insight}", unsafe_allow_html=True)

                                st.write("Specifically:")
                                st.write(f"- Improved Queries: {', '.join(map(str, analysis['improved_queries']))}")
                                st.write(f"- Degraded Queries: {', '.join(map(str, analysis['degraded_queries']))}")
                                st.write(f"- Unchanged Queries: {', '.join(map(str, analysis['unchanged_queries']))}")

st.divider()

# Per query Measure Performance Plots Comparison with a Threshold
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Experimental Evaluation</span> Vs <span style="color:red;">Median/Threshold</span></h3>""",
                unsafe_allow_html=True)
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
                key="me_slider4",
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

            selected_measures = st.multiselect("Select additional measures:", custom_user, default=custom_user[1:2], key="multiselect_4")

        with col2:
            if 'qme_selected_cutoff' not in st.session_state:
                st.session_state.qme_selected_cutoff = 10  # Default cutoff value

            selected_cutoff = st.number_input("Enter cutoff value:", min_value=1, value=10, max_value=1000, step=1, key="cutoff_4")

            # Update session state with current selections
            st.session_state.qme_selected_measures = selected_measures
            st.session_state.qme_selected_cutoff = selected_cutoff

        st.write("""If value is 0.0 (default), it compares the per query retrieval performance of the N experiment to a threshold value equal to the median performance of the selected measures, "
                 "computed based on N-1 experiments.
                 Otherwise, you can select any threshold [0.0,1.0]""")

        st.session_state.qme_comparison_thresh = st.number_input("Select a performance threshold (will be the same across queries):", min_value=0., value=0., max_value=1.0, step=.05,
                                                                 key="cutoff_5")

        results = per_query_evaluation(st.session_state.qme_selected_qrels, st.session_state.qme_selected_runs, st.session_state.qme_selected_measures, st.session_state.qme_relevance_threshold,
                                       st.session_state.qme_selected_cutoff,
                                       None, st.session_state.qme_comparison_thresh)

        # This analysis leverages the median performance of the N-1 experiments
        if st.session_state.qme_comparison_thresh == 0.0:
            analysis_results = analyze_performance_difference_median(results)

            st.write("""<h4><span style="color:red;">Information Related to the Analysis</span></h4>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                This analysis compares each run against the median threshold, identifying:
                1. Percentage of queries improved, degraded, or unchanged
                2. Average and median differences in performance
                3. Variability in performance differences across queries
                """)

            with col2:
                st.markdown("""
                The analysis also provides insights on:
                1. Overall trend of improvement or degradation
                2. Presence of extreme values affecting the results
                3. Variability in performance differences
                """)

            # Create a row for each run
            for run, run_analysis in analysis_results.items():
                st.write(f"""<center><h4>{run} <span style="color:red;">vs</span> Median Threshold</h4></center>""", unsafe_allow_html=True)

                # Calculate the number of measures
                measures = list(run_analysis.keys())
                num_measures = len(measures)
                num_rows = math.ceil(num_measures / 2)

                # Create rows and columns for measures
                for i in range(num_rows):
                    col1, col2 = st.columns(2)

                    for j, col in enumerate([col1, col2]):
                        if i * 2 + j < num_measures:
                            measure = measures[i * 2 + j]
                            analysis = run_analysis[measure]
                            with col:
                                st.subheader(f"Result Analysis based on {measure}")
                                st.write(f"Improved Queries: {len(analysis['improved_queries'])} ({analysis['pct_improved']:.2f}%)")
                                st.write(f"Degraded Queries: {len(analysis['degraded_queries'])} ({analysis['pct_degraded']:.2f}%)")
                                st.write(f"Unchanged Queries: {len(analysis['unchanged_queries'])} ({analysis['pct_unchanged']:.2f}%)")
                                with st.expander("See detailed analysis"):
                                    st.write(f"Average Difference: {analysis['avg_diff']:.3f}")
                                    st.write(f"Median Difference: {analysis['median_diff']:.3f}")
                                    st.write(f"Standard Deviation of Difference: {analysis['std_diff']:.3f}")
                                    st.write('----')
                                    # Additional insights
                                    insights = []
                                    if analysis['pct_improved'] > analysis['pct_degraded']:
                                        insights.append(
                                            f"""This run shows overall improvement over the median for <span style="color:red;">{measure}</span>, with <span style="color:red;">{analysis['pct_improved']:.2f}</span>% of queries improved.""")
                                    elif analysis['pct_improved'] < analysis['pct_degraded']:
                                        insights.append(
                                            f"""This run shows overall degradation compared to the median for <span style="color:red;">{measure}</span>, with <span style="color:red;">{analysis['pct_degraded']:.2f}</span>% of queries degraded.""")
                                    else:
                                        insights.append(f"""This run shows no significant overall difference from the median for <span style="color:red;">{measure}</span>.""")

                                    if analysis['avg_diff'] > 0:
                                        insights.append(f"""The average difference is positive (<span style="color:red;">{analysis['avg_diff']:.3f}</span>), indicating an overall improvement.""")
                                    elif analysis['avg_diff'] < 0:
                                        insights.append(f"""The average difference is negative (<span style="color:red;">{analysis['avg_diff']:.3f}</span>), indicating an overall degradation.""")

                                    if abs(analysis['median_diff']) > abs(analysis['avg_diff']):
                                        insights.append("The median difference is larger than the average, suggesting some extreme values are influencing the results.")

                                    if analysis['std_diff'] > abs(analysis['avg_diff']):
                                        insights.append("""High variability in differences across queries. Some queries may have significantly larger improvements or degradations than others.""")

                                    for insight in insights:
                                        st.write(f"- {insight}", unsafe_allow_html=True)

                                    st.write("Specifically:")
                                    st.write(f"- Queries above Median: {', '.join(map(str, analysis['improved_queries']))}")
                                    st.write(f"- Queries below Median: {', '.join(map(str, analysis['degraded_queries']))}")
                                    st.write(f"- Queries on Median: {', '.join(map(str, analysis['unchanged_queries']))}")
        else:
            # This analysis leverages a threshold selected by the user
            analysis_results = analyze_performance_difference_threshold(results, st.session_state.qme_comparison_thresh)

            st.write("""<h4><span style="color:red;">Information Related to the Analysis</span></h4>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                   This analysis compares each run against the specified threshold, identifying:
                   1. Percentage of queries improved, degraded, or unchanged
                   2. Average and median differences in performance
                   3. Variability in performance differences across queries
                   """)

            with col2:
                st.markdown("""
                   The analysis also provides insights on:
                   1. Overall trend of improvement or degradation
                   2. Presence of extreme values affecting the results
                   3. Variability in performance differences
                   """)

            # Create a row for each run
            for run, run_analysis in analysis_results.items():
                st.write(f"""<center><h4>{run} <span style="color:red;">vs</span> Threshold ({st.session_state.qme_comparison_thresh:.2f})</h4></center>""", unsafe_allow_html=True)

                # Calculate the number of measures
                measures = list(run_analysis.keys())
                num_measures = len(measures)
                num_rows = math.ceil(num_measures / 2)

                # Create rows and columns for measures
                for i in range(num_rows):
                    col1, col2 = st.columns(2)

                    for j, col in enumerate([col1, col2]):
                        if i * 2 + j < num_measures:
                            measure = measures[i * 2 + j]
                            analysis = run_analysis[measure]
                            with col:
                                st.subheader(f"Result Analysis based on {measure}")
                                st.write(f"Improved Queries: {len(analysis['improved_queries'])} ({analysis['pct_improved']:.2f}%)")
                                st.write(f"Degraded Queries: {len(analysis['degraded_queries'])} ({analysis['pct_degraded']:.2f}%)")
                                st.write(f"Unchanged Queries: {len(analysis['unchanged_queries'])} ({analysis['pct_unchanged']:.2f}%)")
                                with st.expander("See detailed analysis"):
                                    st.write(f"Average Difference: {analysis['avg_diff']:.3f}")
                                    st.write(f"Median Difference: {analysis['median_diff']:.3f}")
                                    st.write(f"Standard Deviation of Difference: {analysis['std_diff']:.3f}")
                                    st.write(f"Relative Improvement: {analysis['relative_improvement']:.2f}%")
                                    st.write(f"Effect Size (Cohen's d): {analysis['effect_size']:.3f}")
                                    st.write('----')
                                    # Additional insights
                                    insights = []
                                    if analysis['pct_improved'] > analysis['pct_degraded']:
                                        insights.append(
                                            f"""This run shows overall improvement over the threshold for <span style="color:red;">{measure}</span>, with <span style="color:red;">{analysis['pct_improved']:.2f}</span>% of queries improved.""")
                                    elif analysis['pct_improved'] < analysis['pct_degraded']:
                                        insights.append(
                                            f"""This run shows overall degradation compared to the threshold for <span style="color:red;">{measure}</span>, with <span style="color:red;">{analysis['pct_degraded']:.2f}</span>% of queries degraded.""")
                                    else:
                                        insights.append(f"""This run shows no significant overall difference from the threshold for <span style="color:red;">{measure}</span>.""")

                                    if analysis['avg_diff'] > 0:
                                        insights.append(f"""The average difference is positive (<span style="color:red;">{analysis['avg_diff']:.3f}</span>), indicating an overall improvement.""")
                                    elif analysis['avg_diff'] < 0:
                                        insights.append(f"""The average difference is negative (<span style="color:red;">{analysis['avg_diff']:.3f}</span>), indicating an overall degradation.""")

                                    if abs(analysis['median_diff']) > abs(analysis['avg_diff']):
                                        insights.append("The median difference is larger than the average, suggesting some extreme values are influencing the results.")

                                    if analysis['std_diff'] > abs(analysis['avg_diff']):
                                        insights.append("""High variability in differences across queries. Some queries may have significantly larger improvements or degradations than others.""")

                                    for insight in insights:
                                        st.write(f"- {insight}", unsafe_allow_html=True)

                                    st.write("Specifically:")
                                    st.write(f"- Queries above Threshold: {', '.join(map(str, analysis['improved_queries']))}")
                                    st.write(f"- Queries below Threshold: {', '.join(map(str, analysis['degraded_queries']))}")
                                    st.write(f"- Queries on Threshold: {', '.join(map(str, analysis['unchanged_queries']))}")

st.divider()

# Additional Analysis
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Personal Notes</span></h3>""", unsafe_allow_html=True)

    st.text_area(
        "Please add additional comments regarding this experiment.",
        "",
        key="placeholder",
    )
st.divider()

st.markdown("""<h4 style="text-align:center;"><span style="color:red;">To export the report as PDF press (⌘+P or Ctrl+P)</span></h4>""", unsafe_allow_html=True)
