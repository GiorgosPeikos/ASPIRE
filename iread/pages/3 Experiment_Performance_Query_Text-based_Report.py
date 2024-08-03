import math
from utils.data_handler import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.eval_query_collection import analyze_query_judgements
from utils.eval_query_text_based import *

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Load custom CSS
load_css("css/styles.css")

# Title for the page
st.markdown("""<div style="text-align: center;"><h1>Query Text-based Analysis Across Multiple Experiments<h1></div>""", unsafe_allow_html=True)

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
    st.session_state.qmet_selected_qrels = load_qrel_data(os.path.join(qrels_dir, qrels_file))
    st.session_state.qmet_max_relevance = st.session_state.qmet_selected_qrels["relevance"].max()

    # Load selected runs data
    st.session_state.qmet_selected_runs = {}
    for run_file in selected_runs_files:
        st.session_state.qmet_selected_runs[run_file] = load_run_data(os.path.join(runs_dir, run_file))

    if st.session_state.qmet_selected_runs:
        st.markdown(
            f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{", ".join(selected_runs_files).replace('.txt', '').replace('.csv', '')}</span> experiments using the <span style="color:red;">{qrels_file}</span> qrels.</div>""",
            unsafe_allow_html=True)

    if queries_file:
        st.session_state.qmet_selected_queries = load_query_data(os.path.join(queries_dir, queries_file))
        st.markdown(
            f"""<div style="text-align: center;">This experiment is associated with the <span style="color:red;">{queries_file}</span> queries.</div>""",
            unsafe_allow_html=True)

# Functionality that allows to randomly select queries for analysis, size and queries.
if 'qmet_selected_queries' in st.session_state and not st.session_state.qmet_selected_queries.empty:
    if len(st.session_state.qmet_selected_queries) > 100:
        with st.container():
            st.write("""<h3>Query Sampling - <span style="color:red;">Sampling Queries to Facilitate Experiment Analysis</span></h3>""", unsafe_allow_html=True)

            # Initialize random_state in session state if it doesn't exist
            if 'qmet_random_state' not in st.session_state:
                st.session_state.qmet_random_state = 42
            if 'qmet_random_size' not in st.session_state:
                st.session_state.qmet_random_size = 100

            st.write(f"""<div style="text-align: center;">  ⚠️ Note: Too many available queries (<span style="color:red;">{len(st.session_state.qmet_selected_queries)}</span>).
            To enhance the following analysis, a random set will be used. Please select the following:</div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.session_state.qmet_random_size = st.number_input(
                    "Set Number of queries to be randomly selected (default: 49)",
                    min_value=1,
                    value=st.session_state.qmet_random_size,
                    max_value=len(st.session_state.qmet_selected_queries),
                    step=1
                )
            with col2:
                st.session_state.qmet_random_state = st.number_input(
                    "Set random state for query selection (default: 42)",
                    min_value=1,
                    value=st.session_state.qmet_random_state,
                    max_value=100,
                    step=1
                )

            # Main content logic
            st.write(f"""<div style="text-align: center;"> A total of <span style="color:red;">{st.session_state.qmet_random_size}</span> random queries have been
            selected based on a random state equal to <span style="color:red;">{st.session_state.qmet_random_state}</span> and will be used for the upcoming analyses.</div>""", unsafe_allow_html=True)

            st.session_state.qmet_selected_queries_random = st.session_state.qmet_selected_queries.sample(n=st.session_state.qmet_random_size, random_state=st.session_state.qmet_random_state)

            st.write(f"""<div style="text-align: center;"> Number of randomly selected queries that would be used for analysis: <span style="color:red;"
                >{len(st.session_state.qmet_selected_queries_random)}</span></div>""", unsafe_allow_html=True)

            query_ids = np.sort(st.session_state.qmet_selected_queries_random.query_id.values)
            query_ids_str = ", ".join(map(str, query_ids))
            st.write(f"Selected Query (IDs): {query_ids_str}")

            st.divider()
    else:
        if 'qmet__selected_queries_random' not in st.session_state:
            st.session_state.qmet_selected_queries_random = st.session_state.qmet_selected_queries
            st.write(f"""<div style="text-align: center;"> All <span style="color:red;">{len(st.session_state.qmet_selected_queries_random)}</span> provided queries will be used for the 
            following analyses.</div>""", unsafe_allow_html=True)

        st.divider()

# Per query Relevance Judgements
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Relevance Judgments per Query</span></h3>""", unsafe_allow_html=True)

    if 'qmet_selected_runs' not in st.session_state:
        st.warning("Please select a set of queries to begin your evaluation.", icon="⚠")
    else:
        results = plot_query_relevance_judgements(st.session_state.qmet_selected_qrels)
        analysis_results = analyze_query_judgements(results)

        relevance_labels = [label for label in analysis_results['label_comparison'] if label != 'combined']
        num_columns = len(relevance_labels)
        columns = st.columns(num_columns)

        for i, label in enumerate(relevance_labels):
            with columns[i]:
                color = 'red' if label == 'Relevance_Label_0' else 'blue'
                label_name = 'Irrelevant' if label == 'Relevance_Label_0' else label
                st.markdown(f"<h4><span style='color:{color};'>{label_name}</span></h4>", unsafe_allow_html=True)

                comparison = analysis_results['label_comparison'][label]

                with st.expander("Easy Queries"):
                    st.write(", ".join(map(str, comparison['easy_queries'][::])))
                    st.markdown("*These queries have equal or higher relevant documents compared to irrelevant ones.*")

                with st.expander("Hard Queries"):
                    st.write(", ".join(map(str, comparison['hard_queries'][::])))
                    st.markdown("*These queries have very few relevant documents compared to irrelevant ones.*")

                with st.expander("Min/Max Queries"):
                    st.write(f"**Min:** {comparison['min_query']}")
                    st.write(f"**Max:** {comparison['max_query']}")
                    st.markdown("*Queries with the minimum and maximum number of relevant documents for this label.*")

        st.markdown("### Combined Relevance Labels")
        with st.expander("See Analysis"):
            combined = analysis_results['label_comparison']['combined']
            st.write("**Easy Queries:** " + ", ".join(map(str, combined['easy_queries'][::])))
            st.write("**Hard Queries:** " + ", ".join(map(str, combined['hard_queries'][::])))
            st.write(f"**Min Query:** {combined['min_query']}")
            st.write(f"**Max Query:** {combined['max_query']}")
            st.markdown("""
            This analysis combines all relevance labels (except irrelevant) and compares them to the irrelevant label.
            - Easy queries have more relevant documents (across all labels) than irrelevant ones.
            - Hard queries have significantly fewer relevant documents than irrelevant ones.
            - Min and Max queries have the least and most relevant documents respectively, considering all relevance labels.
            """)

st.divider()

# Query Performance vs Query length
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Query Length vs Query Performance</span></h3>""", unsafe_allow_html=True)
    _, _, custom_user, default_measures, _, _ = return_available_measures()

    if 'qmet_selected_runs' not in st.session_state:
        st.warning("Please select a set of queries to begin your evaluation.", icon="⚠")
    else:
        with st.expander("See Details and Interpretations"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("""
                ### Visualization Explanation
                1. **Scatter Plot with Moving Average**:
                   - Each point represents a single query.
                   - X-axis: Token length of the query.
                   - Y-axis: Performance measure (e.g., AP@10, nDCG@10).
                   - Color gradient: Indicates performance.
                   - Red line: Moving average of performance.
    
                2. **Moving Average**:
                   - Smooths out short-term fluctuations and highlights longer-term trends.
                   - Calculated using a window of 20 queries.
                   - Helps visualize how performance generally changes with token length.
                   - Upward trend: Performance tends to improve with token length.
                   - Downward trend: Performance tends to decrease with token length.
                   - Flat line: No strong relationship between token length and performance.
    
                ### Bucketing Methods
                This analysis compares two methods of grouping queries based on their token lengths:
                1. **Equal-width buckets**: Each bucket represents an equal range of token lengths.
                2. **Equal-frequency buckets**: Each bucket contains approximately the same number of queries.
                """)

            with col2:
                st.write("""
                ### Key Considerations
                - **Scatter Plot Interpretation**:
                  - Look for clusters or patterns in the distribution of points.
                  - Identify any outliers.
                  - Compare individual query performance to the moving average trend.
    
                - **Equal-width buckets**:
                  - Show the distribution of performance across the full range of token lengths.
                  - May have varying numbers of queries per bucket!
                  - Useful for identifying performance trends in specific token length ranges.
    
                - **Equal-frequency buckets**:
                  - Ensure a consistent sample size per bucket.
                  - May obscure the actual distribution of token lengths.
                  - Useful for comparing performance across equal-sized query groups.
    
                - Compare both bucketing methods to get a comprehensive understanding of the relationship between token length and performance.
                - Pay attention to the number of queries in each bucket (written in each bar), especially for equal-width buckets.
                - Consider how the bucket analysis results compare to the trends observed in the scatter plot and moving average.
                """)

        # Get the list of selected run files
        selected_runs_files = list(st.session_state.qmet_selected_runs.keys())

        if st.session_state.qmet_max_relevance >= 2:
            st.session_state.qmet_relevance_threshold = st.slider(
                "Select from the Available Relevance Thresholds (Slide)",
                min_value=1,
                max_value=2,
                value=1,
                key="me_slider3",
            )

            if 'qmet_prev_relevance_threshold' not in st.session_state:
                st.session_state.qmet_prev_relevance_threshold = 1

            if st.session_state.qmet_relevance_threshold != st.session_state.qmet_prev_relevance_threshold:
                st.session_state.qmet_prev_relevance_threshold = st.session_state.qmet_relevance_threshold
        else:
            st.session_state.qmet_relevance_threshold = 1
            st.write("""**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""", unsafe_allow_html=True)

            # Create columns
        col1, col2 = st.columns(2)  # Adjust the column width ratio as needed

        with col1:

            # Initialize session state variables if they don't exist
            if 'qmet_selected_measures' not in st.session_state:
                st.session_state.qmet_selected_measures = custom_user[0:4]  # Default selected measures

            selected_measures = st.multiselect("Select additional measures:", custom_user, default=custom_user[1:2])

        with col2:
            if 'qmet_selected_cutoff' not in st.session_state:
                st.session_state.qmet_selected_cutoff = 10  # Default cutoff value

            selected_cutoff = st.number_input("Enter cutoff value:", min_value=1, value=10, max_value=1000, step=1)

            # Update session state with current selections
            st.session_state.qmet_selected_measures = selected_measures
            st.session_state.qmet_selected_cutoff = selected_cutoff

        if len(st.session_state.qmet_selected_measures) >= 1:
            results = per_query_length_evaluation(st.session_state.qmet_selected_qrels, st.session_state.qmet_selected_runs,
                                                  st.session_state.qmet_selected_queries_random,
                                                  st.session_state.qmet_selected_measures,
                                                  st.session_state.qmet_relevance_threshold,
                                                  st.session_state.qmet_selected_cutoff,
                                                  None, None)

