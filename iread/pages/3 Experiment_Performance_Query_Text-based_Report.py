import math
from utils.data_handler import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.eval_per_query import *
from utils.eval_query_collection import analyze_query_judgements

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
        if 'qme_selected_runs' not in st.session_state:
            st.session_state.qme_selected_queries_random = st.session_state.qme_selected_queries
            st.write(f"""<div style="text-align: center;"> All <span style="color:red;">{len(st.session_state.qme_selected_queries_random)}</span> provided queries will be used for the 
            following analyses.</div>""", unsafe_allow_html=True)

        st.divider()


# Per query Measure Performance Plots

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

                                col1in, col2in = st.columns([1, 2])

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

st.divider()

