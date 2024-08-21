import streamlit as st
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed")
import numpy as np
from utils.ui import load_css
from utils.data_handler import *
from utils.eval_core import return_available_measures
from utils.eval_query_collection import analyze_query_judgements
from utils.plots import plot_query_relevance_judgements, plot_query_performance_vs_query_length, create_relevance_wordclouds, plot_performance_similarity
from utils.eval_query_text_based import per_query_length_evaluation

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
        if 'qmet_selected_queries_random' not in st.session_state:
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
        # st.session_state.qmet_query_rel_judg --> Usage later on the code --> Contains for each query its relevance assessments, in dictionary format:
        # "2":{
        #   "irrelevant": 478,
        #   "relevant": {
        #     "Relevance_Label_1": 36,
        #     "Relevance_Label_2": 11
        #   }
        # }
        st.session_state.qmet_query_rel_judg = plot_query_relevance_judgements(st.session_state.qmet_selected_qrels)
        analysis_results = analyze_query_judgements(st.session_state.qmet_query_rel_judg)

        relevance_labels = [label for label in analysis_results['label_comparison'] if label != 'combined']
        num_columns = len(relevance_labels)
        columns = st.columns(num_columns)

        for i, label in enumerate(relevance_labels):
            with columns[i]:
                color = 'red' if label == 'Relevance_Label_0' else 'blue'
                label_name = 'Irrelevant' if label == 'Relevance_Label_0' else label
                st.markdown(f"<h5><span style='color:{color};'>{label_name}</span></h5>", unsafe_allow_html=True)

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

        st.markdown("###### Combined Relevance Labels")
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
        st.markdown("###### Manually Examine Sampled Queries")
        with st.expander("See Queries"):
            st.dataframe(st.session_state.qmet_selected_queries_random[['query_id', 'query_text']], use_container_width=True, hide_index=True)

st.divider()

# Query Text Analysis based on Relevance Judgements
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Query Text Analysis based on their Relevance Judgements</span></h3>""", unsafe_allow_html=True)
    _, _, custom_user, default_measures, _, _ = return_available_measures()

    if 'qmet_selected_runs' not in st.session_state:
        st.warning("Please select a set of queries to begin your evaluation.", icon="⚠")

    else:
        with st.expander("Details of the Query Selection (Outlier Detection) Methods"):
            st.markdown("""
            ### Median Absolute Deviation (MAD)
            The MAD method uses the median of the absolute deviations from the median of the data. It's more robust against extreme outliers than methods based on mean and standard deviation.

            - **Pros**: Robust against extreme outliers, works well with skewed distributions.
            - **Cons**: May be less sensitive to subtle outliers in some cases.
            - **When to use**: When your data may contain extreme outliers or is not normally distributed.
            - **Threshold effect**: A higher threshold will result in fewer queries being classified as outliers. As you increase the threshold, only the most extreme values will be considered outliers.

            ### Percentile-based method
            This method defines outliers based on the data's percentiles. It considers values below the 5th percentile or above the 95th percentile as outliers.

            - **Pros**: Simple to understand, always identifies a fixed proportion of the data as outliers.
            - **Cons**: May not adapt well to different data distributions.
            - **When to use**: When you want a straightforward method that always identifies a certain percentage of data points as outliers.
            - **Threshold effect**: This method doesn't use a user-defined threshold. It always selects a fixed percentage of queries as outliers based on the predefined percentiles (5th and 95th).

            ### Modified Z-score
            Similar to the standard Z-score, but uses median and MAD instead of mean and standard deviation. This makes it more robust against outliers.

            - **Pros**: More robust than standard Z-score, works well for skewed distributions.
            - **Cons**: May be overly sensitive for very small datasets.
            - **When to use**: When you want a balance between robustness and sensitivity to outliers.
            - **Threshold effect**: The threshold determines how many standard deviations (calculated using MAD) a value needs to be from the median to be considered an outlier. A higher threshold will result in fewer outliers, focusing on more extreme values.

            ### Max-Min-Median
            This method simply selects the top 5 queries with the most relevance assessments, the bottom 5, and 5 queries around the median number of relevance assessments.

            - **Pros**: Straightforward, always provides a fixed number of queries for each category.
            - **Cons**: Doesn't adapt to the data distribution, may not capture true outliers in all cases.
            - **When to use**: When you want a quick overview of the extreme values and median in your dataset.
            - **Threshold effect**: This method doesn't use a threshold. It always selects the same number of queries regardless of the data distribution.

            ### General Note on Thresholds
            For methods that use thresholds (MAD, Modified Z-score), increasing the threshold makes the outlier detection more conservative, resulting in fewer queries being classified as outliers. Decreasing the threshold makes the detection more sensitive, potentially identifying more queries as outliers. The optimal threshold often depends on your specific data and use case.
            """)

        col1, col2 = st.columns(2)

        with col1:
            # Method selection with 'Max-Min' as the default
            st.session_state.qmet_method_text_plots = st.selectbox(
                'Select outlier detection method to sample queries. By default, **5 queries** with the most, least, and around median relevance assessments are selected.',
                ['Max-Min-Median', 'Median Absolute Deviation', 'Percentile-based method', 'Modified Z-score'],
                index=0  # This sets 'Max-Min' as the default
            )

        with col2:
            # Threshold selection (not applicable for 'Percentile-based method' and 'Max-Min')
            st.session_state.qmet_threshold_text_plots = None
            if st.session_state.qmet_method_text_plots not in ['Percentile-based method', 'Max-Min-Median']:
                st.session_state.qmet_threshold_text_plots = st.slider('Select threshold', 1.0, 10.0, 1.5, 0.1)

        # Run analysis button
        create_relevance_wordclouds(
            st.session_state.qmet_query_rel_judg,
            st.session_state.qmet_selected_queries_random,
            method=st.session_state.qmet_method_text_plots,
            threshold=st.session_state.qmet_threshold_text_plots)

st.divider()


# Query Performance vs Query length
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Query Performance based on  Query Length</span></h3>""", unsafe_allow_html=True)
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

            selected_cutoff = st.number_input("Enter measure cutoff value:", min_value=1, value=10, max_value=1000, step=1)

            # Update session state with current selections
            st.session_state.qmet_selected_measures = selected_measures
            st.session_state.qmet_selected_cutoff = selected_cutoff

        if len(st.session_state.qmet_selected_measures) >= 1:
            st.session_state.per_query_length_res = per_query_length_evaluation(st.session_state.qmet_selected_qrels, st.session_state.qmet_selected_runs,
                                                                                st.session_state.qmet_selected_queries_random,
                                                                                st.session_state.qmet_selected_measures,
                                                                                st.session_state.qmet_relevance_threshold,
                                                                                st.session_state.qmet_selected_cutoff,
                                                                                None, None)

            # Call the plot function per measure scores
            plot_query_performance_vs_query_length(st.session_state.per_query_length_res)

st.divider()


# Query Performance vs Query Similarity
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Query Performance vs Query Text Similarity</span></h3>""", unsafe_allow_html=True)

    if 'qmet_selected_runs' not in st.session_state:
        st.warning("Please select a set of queries to begin your evaluation.", icon="⚠")
    else:
        with st.expander('See Details and Interpretations'):
            st.write("""
            This visualization shows queries in a 3D space based on their semantic similarity.
            - Each point represents a query.
            - The position of the point is determined by the query's embedding, reduced to 3 dimensions using t-SNE.
            - The color of the point represents the performance measure for that query.
            - Use the dropdown to select different performance measures and runs for coloring the points.
            """)

        # Get the list of selected run files
        selected_runs_files = list(st.session_state.qmet_selected_runs.keys())

        if st.session_state.qmet_max_relevance >= 2:
            st.session_state.qmet_relevance_threshold = st.slider(
                "Select from the Available Relevance Thresholds (Slide)",
                min_value=1,
                max_value=2,
                value=1,
                key="me_slider4",
            )

            if 'qmet_prev_relevance_threshold' not in st.session_state:
                st.session_state.qmet_prev_relevance_threshold = 1

            if st.session_state.qmet_relevance_threshold != st.session_state.qmet_prev_relevance_threshold:
                st.session_state.qmet_prev_relevance_threshold = st.session_state.qmet_relevance_threshold
        else:
            st.session_state.qmet_relevance_threshold = 1
            st.write("""**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)  # Adjust the column width ratio as needed

        with col1:

            # Initialize session state variables if they don't exist
            if 'qmet_selected_measures' not in st.session_state:
                st.session_state.qmet_selected_measures = custom_user[0:4]  # Default selected measures

            selected_measures = st.multiselect("Select additional measures:", custom_user, default=custom_user[1:2], key='select_measures2')

        with col2:
            if 'qmet_selected_cutoff' not in st.session_state:
                st.session_state.qmet_selected_cutoff = 10  # Default cutoff value

            selected_cutoff = st.number_input("Enter measure cutoff value:", min_value=1, value=10, max_value=1000, step=1, key='cutoff2')

            # Update session state with current selections
            st.session_state.qmet_selected_measures = selected_measures
            st.session_state.qmet_selected_cutoff = selected_cutoff

        st.session_state.emb_model_name = st.text_input(
            "**Enter a HuggingFace model name that will be used to estimate the query embeddings. Default model:**",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        plot_performance_similarity(
            st.session_state.qmet_selected_queries_random,
            st.session_state.qmet_selected_qrels,
            st.session_state.qmet_selected_runs,
            st.session_state.qmet_selected_measures,
            st.session_state.qmet_selected_cutoff,
            st.session_state.qmet_relevance_threshold,
            st.session_state.emb_model_name
        )

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

st.markdown("""<h5 style="text-align:center;"><span style="color:red;">To export the report as PDF press (⌘+P or Ctrl+P)</span></h5>""", unsafe_allow_html=True)
