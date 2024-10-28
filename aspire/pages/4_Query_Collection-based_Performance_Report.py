import os

import numpy as np
import streamlit as st
from utils.data_handler import load_qrel_data, load_query_data, load_run_data
from utils.eval_query_collection import (
    analyze_query_judgements, display_further_details_multi_query_docs,
    documents_retrieved_by_experiments, find_multi_query_docs,
    find_ranked_pos_of_multi_query_docs)
from utils.plots import (plot_documents_retrieved_by_experiments,
                         plot_multi_query_docs,
                         plot_query_relevance_judgements,
                         plot_rankings_docs_rel_ids)
from utils.ui import load_css

st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load custom CSS
load_css("aspire/css/styles.css")

# Title for the page
st.markdown(
    """<div style="text-align: center;"><h1>Retrieval Evaluation Report based on the Retrieved Documents<h1></div>""",
    unsafe_allow_html=True,
)

# Container for loading data
with st.container():
    st.markdown(
        "<h3>Loading Queries, Qrels, Runs for Analysis</h3>", unsafe_allow_html=True
    )

    with st.expander("See Details"):
        st.subheader("Overview")
        st.write(
            "This section allows you to load and select the necessary data files for your retrieval evaluation."
        )
        cola, colb = st.columns(2)
        with cola:
            st.subheader("How it works")
            st.write(
                "The system loads query files, qrel (relevance judgment) files, and retrieval run files from predefined directories. You can select multiple run files for comparison."
            )
        with colb:
            st.subheader("How to use")
            st.write(
                "Use this section to set up your evaluation environment. Ensure you have the correct files selected to get accurate and meaningful results in the subsequent analyses."
            )

    columns = st.columns([1, 1, 3])

    # Select Queries
    with columns[0]:
        queries_dir = "retrieval_experiments/queries"
        queries_file = st.selectbox(
            "Select Queries",
            os.listdir(queries_dir) if os.path.exists(queries_dir) else [],
        )

    # Select Qrels
    with columns[1]:
        qrels_dir = "retrieval_experiments/qrels"
        qrels_file = st.selectbox(
            "Select Qrels", os.listdir(qrels_dir) if os.path.exists(qrels_dir) else []
        )

    # Select Retrieval Runs (multiselect)
    with columns[2]:
        runs_dir = "retrieval_experiments/retrieval_runs"
        selected_runs_files = st.multiselect(
            "Select Retrieval Runs",
            os.listdir(runs_dir) if os.path.exists(runs_dir) else [],
        )

# Button to start evaluation
if st.button("Begin the Experimental Evaluation!", key="qmed_stButtonCenter"):

    if not qrels_file:
        st.write("Please select Qrels file to proceed.")
        st.stop()

    if not selected_runs_files:
        st.write("Please select at least one Retrieval Run file to proceed.")
        st.stop()

    # Load Qrels data
    st.session_state.qmed_selected_qrels = load_qrel_data(
        os.path.join(qrels_dir, qrels_file)
    )
    st.session_state.qmed_max_relevance = st.session_state.qmed_selected_qrels[
        "relevance"
    ].max()

    # Load selected runs data
    st.session_state.qmed_selected_runs = {}
    for run_file in selected_runs_files:
        st.session_state.qmed_selected_runs[run_file] = load_run_data(
            os.path.join(runs_dir, run_file)
        )

    if st.session_state.qmed_selected_runs:
        st.markdown(
            f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{", ".join(selected_runs_files).replace('.txt', '').replace('.csv', '')}</span> experiments using the <span style="color:red;">{qrels_file}</span> qrels.</div>""",
            unsafe_allow_html=True,
        )

    if queries_file:
        st.session_state.qmed_selected_queries = load_query_data(
            os.path.join(queries_dir, queries_file)
        )
        st.markdown(
            f"""<div style="text-align: center;">This experiment is associated with the <span style="color:red;">{queries_file}</span> queries.</div>""",
            unsafe_allow_html=True,
        )

st.divider()

# Functionality that allows to randomly select queries for analysis, size and queries.
if (
    "qmed_selected_queries" in st.session_state
    and not st.session_state.qmed_selected_queries.empty
):
    if len(st.session_state.qmed_selected_queries) > 500:
        with st.container():
            st.write(
                """<h3>Query Sampling - <span style="color:red;">Sampling Queries to Facilitate Experiment Analysis</span></h3>""",
                unsafe_allow_html=True,
            )

            # Initialize random_state in session state if it doesn't exist
            if "qmed_random_state" not in st.session_state:
                st.session_state.qmed_random_state = 42
            if "qmed_random_size" not in st.session_state:
                st.session_state.qmed_random_size = 250

            st.write(
                f"""<div style="text-align: center;">  ⚠️ Note: Too many available queries (<span style="color:red;">{len(st.session_state.qmed_selected_queries)}</span>).
            To enhance the following analysis, a random set will be used. Please select the following:</div>""",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.session_state.qmed_random_size = st.number_input(
                    "Set Number of queries to be randomly selected (default: 49)",
                    min_value=1,
                    value=st.session_state.qmed_random_size,
                    max_value=len(st.session_state.qmed_selected_queries),
                    step=1,
                )
            with col2:
                st.session_state.qmed_random_state = st.number_input(
                    "Set random state for query selection (default: 42)",
                    min_value=1,
                    value=st.session_state.qmed_random_state,
                    max_value=100,
                    step=1,
                )

            # Main content logic
            st.write(
                f"""<div style="text-align: center;"> A total of <span style="color:red;">{st.session_state.qmed_random_size}</span> random queries have been
            selected based on a random state equal to <span style="color:red;">{st.session_state.qmed_random_state}</span> and will be used for the upcoming analyses.</div>""",
                unsafe_allow_html=True,
            )

            st.session_state.qmed_selected_queries_random = (
                st.session_state.qmed_selected_queries.sample(
                    n=st.session_state.qmed_random_size,
                    random_state=st.session_state.qmed_random_state,
                )
            )

            st.write(
                f"""<div style="text-align: center;"> Number of randomly selected queries that would be used for analysis: <span style="color:red;"
                >{len(st.session_state.qmed_selected_queries_random)}</span></div>""",
                unsafe_allow_html=True,
            )

            query_ids = np.sort(
                st.session_state.qmed_selected_queries_random.query_id.values
            )
            query_ids_str = ", ".join(map(str, query_ids))
            st.write(f"Selected Query (IDs): {query_ids_str}")

            st.divider()
    else:
        st.session_state.qmed_selected_queries_random = (
            st.session_state.qmed_selected_queries
        )
        # if "qmed_selected_queries_random" in st.session_state:
        #     st.write(
        #         f"""<div style="text-align: center;"> All <span style="color:red;">{len(st.session_state.qmed_selected_queries_random)}</span> provided queries will be used for the
        #         following analyses.</div>""",
        #         unsafe_allow_html=True,
        #     )
        st.divider()

# Per query Relevance Judgements
with st.container():
    st.markdown(
        """<h3>Document Collection - <span style="color:red;">Relevance Judgments per Query</span></h3>""",
        unsafe_allow_html=True,
    )

    with st.expander("See Analysis Details and Interpretations"):
        st.subheader("Overview")
        st.write(
            "This analysis showcases for each query its number of relevance judgments."
        )
        cola, colb = st.columns(2)
        with cola:
            st.subheader("How it works")
            st.write(
                "<b>The analysis leverages only the provided Qrels file!</b>",
                unsafe_allow_html=True,
            )
        with colb:
            st.subheader("How to use")
            st.write(
                "Use this section in combination with the following to understand how a query's relevance judgments might impact its performance."
            )

    if "qmed_selected_runs" not in st.session_state:
        st.warning("Please select a set of queries to begin your evaluation.", icon="⚠")
    else:
        # st.session_state.qmed_query_rel_judg --> Usage later on the code --> Contains for each query its relevance assessments, in dictionary format:
        # "2":{
        #   "irrelevant": 478,
        #   "relevant": {
        #     "Relevance_Label_1": 36,
        #     "Relevance_Label_2": 11
        #   }
        # }
        st.session_state.qmed_query_rel_judg = plot_query_relevance_judgements(
            st.session_state.qmed_selected_qrels
        )
        analysis_results = analyze_query_judgements(
            st.session_state.qmed_query_rel_judg
        )

        relevance_labels = [
            label
            for label in analysis_results["label_comparison"]
            if label != "combined"
        ]
        num_columns = len(relevance_labels)
        columns = st.columns(num_columns)

        for i, label in enumerate(relevance_labels):
            with columns[i]:
                color = "red" if label == "Relevance_Label_0" else "blue"
                label_name = "Irrelevant" if label == "Relevance_Label_0" else label
                st.markdown(
                    f"<h5><span style='color:{color};'>{label_name}</span></h5>",
                    unsafe_allow_html=True,
                )

                comparison = analysis_results["label_comparison"][label]

                with st.expander("Easy Queries"):
                    st.write(", ".join(map(str, comparison["easy_queries"][::])))
                    st.markdown(
                        "*These queries have equal or higher relevant documents compared to irrelevant ones.*"
                    )

                with st.expander("Hard Queries"):
                    st.write(", ".join(map(str, comparison["hard_queries"][::])))
                    st.markdown(
                        "*These queries have very few relevant documents compared to irrelevant ones.*"
                    )

                with st.expander("Min/Max Queries"):
                    st.write(f"**Min:** {comparison['min_query']}")
                    st.write(f"**Max:** {comparison['max_query']}")
                    st.markdown(
                        "*Queries with the minimum and maximum number of relevant documents for this label.*"
                    )

        st.markdown("###### Combined Relevance Labels")
        with st.expander("See Analysis"):
            combined = analysis_results["label_comparison"]["combined"]
            st.write(
                "**Easy Queries:** " + ", ".join(map(str, combined["easy_queries"][::]))
            )
            st.write(
                "**Hard Queries:** " + ", ".join(map(str, combined["hard_queries"][::]))
            )
            st.write(f"**Min Query:** {combined['min_query']}")
            st.write(f"**Max Query:** {combined['max_query']}")
            st.markdown(
                """
            This analysis combines all relevance labels (except irrelevant) and compares them to the irrelevant label.
            - Easy queries have more relevant documents (across all labels) than irrelevant ones.
            - Hard queries have significantly fewer relevant documents than irrelevant ones.
            - Min and Max queries have the least and most relevant documents respectively, considering all relevance labels.
            """
            )
        st.markdown("###### Manually Examine Sampled Queries")
        with st.expander("See Queries"):
            st.dataframe(
                st.session_state.qmed_selected_queries_random[
                    ["query_id", "query_text"]
                ],
                use_container_width=True,
                hide_index=True,
            )

st.divider()


# Documents that have relevance assessment for more than one query
with st.container():
    st.markdown(
        """<h3>Document Collection - <span style="color:red;">Documents with Relevance Judgments for Multiple Queries</span></h3>""",
        unsafe_allow_html=True,
    )

    with st.expander("See Analysis Details and Interpretations"):
        col1, col2 = st.columns(2)

        st.subheader("Overview")
        st.write(
            """
            This analysis identifies and examines documents that have received relevance assessments for more than one query. These documents are particularly important because they help reveal 
            patterns in how different documents are assessed w.r.t to different queries.
            """
        )
        cola, colb = st.columns(2)
        with cola:
            st.subheader("How it works")
            st.write(
                "<b>The analysis leverages only the provided Qrels file and is based on all of the available Queries in the collection!</b>",
                unsafe_allow_html=True,
            )
        with colb:
            st.subheader("How to use")
            st.write(
                """
            - **Document Relevance Across Queries:** It highlights how the same document might be relevant to multiple queries, indicating the versatility or generality of the document's content.
            - **Query Overlap and Distinction:** Analyzing shared documents between queries helps understand the overlap or distinction in query intent and potential retrieval performance.
            """
            )

    if "qmed_selected_qrels" in st.session_state:
        multi_query_docs = find_multi_query_docs(st.session_state.qmed_selected_qrels)

        if not multi_query_docs.empty:
            st.write(
                f"There are <span style='color:red;'>{len(multi_query_docs)}</span> documents whose relevance have been assessed w.r.t. multiple queries.",
                unsafe_allow_html=True,
            )

            st.session_state.qmed_num_docs = st.slider(
                "Number of documents to display in the graph",
                min_value=1,
                max_value=len(multi_query_docs),
                value=min(100, len(multi_query_docs)),
            )

            # Show the plot
            fig = plot_multi_query_docs(
                multi_query_docs.head(st.session_state.qmed_num_docs)
            )
            st.plotly_chart(fig, use_container_width=True)

            search_doc = st.text_input("Search for a specific document ID")
            if search_doc:
                filtered_docs = multi_query_docs[
                    multi_query_docs.doc_id.str.contains(str(search_doc), case=False)
                ]
                if not filtered_docs.empty:
                    st.dataframe(filtered_docs, hide_index=True)
                else:
                    st.write("No matching documents found.")

            with st.expander("Further Details"):
                display_further_details_multi_query_docs(
                    multi_query_docs, st.session_state.qmed_num_docs
                )

            with st.expander(
                "What was the ranking position and relevance judgment of these documents across experiments and queries?"
            ):
                find_ranked_pos_of_multi_query_docs(
                    multi_query_docs,
                    st.session_state.qmed_num_docs,
                    st.session_state.qmed_selected_runs,
                    st.session_state.qmed_selected_qrels,
                )

        else:
            st.write(
                "No documents found with relevance assessments for multiple queries."
            )
    else:
        st.warning("Please load qrels data to perform this analysis.", icon="⚠")

st.divider()


# Documents that have been retrieved by all systems.
with st.container():
    st.markdown(
        """<h3>Document Collection - <span style="color:red;">Documents retrieved per query by 1, 2, 3, 5, and by all Experiments</span></h3>""",
        unsafe_allow_html=True,
    )

    with st.expander("See Analysis Details and Interpretations"):
        st.write("**Overview**")

        st.write(
            """
        This analysis measures how documents are retrieved across multiple systems for each query. 
        We identify which documents have been retrieved by only 1, 2, 3, 5 (where applicable), at least half+1, 
        and all systems for each query. This helps us understand the consistency and uniqueness of document retrieval 
        across different systems and assess query difficulty.
        """
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**How Calculations Are Made**")
            st.write("- We combine all query-document pairs from all experiments.")
            st.write("- We count how many experiments retrieved each unique pair.")
            st.write(
                "- We calculate the number of pairs retrieved by exactly 1, 2, 3, 5 experiments (where applicable)."
            )
            st.write(
                "- We identify pairs retrieved by at least half+1 of the experiments."
            )
            st.write("- We count pairs retrieved by all experiments.")
            st.write(
                "- Percentages are calculated relative to the total number of unique pairs."
            )
            st.write(
                "- For query difficulty, we calculate the proportion of documents retrieved by only one experiment for each query."
            )

            st.write("**Additional Analysis Features**")
            st.write(
                "- Query Difficulty Ranking: All queries are ranked from most difficult to easiest based on unique retrievals."
            )
            st.write(
                "- Per-Query Document Analysis: For each query, you can view documents retrieved by all systems vs. those retrieved by only one or two systems."
            )
            st.write(
                "- Example Pairs: For each retrieval threshold, example document-query pairs are provided for closer examination."
            )

        with col2:
            st.write("**Interpretation and Use of Results**")
            st.write(
                "- Pairs retrieved by 1 experiment: Unique to specific approaches, might represent noise or novel findings."
            )
            st.write(
                "- Pairs retrieved by multiple experiments: Higher agreement suggests more reliable results."
            )
            st.write(
                "- Pairs retrieved by half+1 experiments: Represent a 'majority vote', considered more robust findings."
            )
            st.write(
                "- Pairs retrieved by all experiments: Most consistent results across all approaches."
            )
            st.write(
                "- Distribution across thresholds: Helps assess overall agreement between different experimental approaches."
            )
            st.write(
                "- Query difficulty: Higher percentages indicate more difficult queries with more unique retrievals."
            )
            st.write(
                "- These metrics guide further investigation, help in ensemble methods, and identify areas of disagreement."
            )

            st.write("**Practical Applications**")
            st.write(
                "- Identify challenging queries that might need refinement or further investigation."
            )
            st.write(
                "- Recognize patterns in query difficulty related to query characteristics or retrieval method strengths/weaknesses."
            )
            st.write(
                "- Assess the overall performance and agreement of retrieval systems across different types of queries."
            )
            st.write(
                "- Guide the development of ensemble methods by understanding where systems agree or disagree."
            )
            st.write(
                "- Inform decisions on which documents to include in a final result set based on retrieval consistency."
            )

    if "qmed_selected_runs" in st.session_state:
        if len(st.session_state.qmed_selected_runs) > 1:
            initial_result_df = documents_retrieved_by_experiments(
                st.session_state.qmed_selected_runs
            )

            # Create a multiselect for excluding runs with a safeguard
            all_runs = list(set(",".join(initial_result_df["run"]).split(",")))
            st.write("<h5>General Analysis</h5>", unsafe_allow_html=True)

            if len(all_runs) > 1:
                max_selectable = len(all_runs) - 2
                excluded_runs = st.multiselect(
                    "Select runs to exclude (at least two must remain):",
                    all_runs,
                    max_selections=max_selectable,
                )

                if len(excluded_runs) == len(all_runs):
                    st.warning(
                        "You must keep at least one experiment. The last selected experiment will be included."
                    )
                    excluded_runs = excluded_runs[:-1]
            else:
                st.info("Only one experiment available. No exclusion possible.")
                excluded_runs = []

            plot_documents_retrieved_by_experiments(initial_result_df, excluded_runs)
    else:
        st.warning(
            "Not sufficient experiments have been selected. Please select at least 2 experiments first.",
            icon="⚠",
        )

st.divider()


# The box like plot for the top positions showing how the documents are ranked.
with st.container():
    st.markdown(
        """<h3>Document Collection - <span style="color:red;">Retrieved Documents, Relevance, Ranking Position</span></h3>""",
        unsafe_allow_html=True,
    )

    with st.expander("See Analysis Details and Interpretations"):
        st.subheader("Overview")
        st.write(
            "This analysis visualizes the relevance of retrieved documents across different ranking positions for multiple experiments."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("How it works")
            st.write(
                """
            - Each row represents a ranking position (1 to selected depth).
            - Each column represents a unique query.
            - Colors indicate the relevance of documents:
              - Red: Low relevance
              - Orange: Medium relevance
              - Green: High relevance
              - Gray: Unjudged relevance
            - Multiple experiments are shown side by side for comparison.
            """
            )

        with col2:
            st.subheader("How to use")
            st.write(
                """
            - Hover over cells to see detailed information.
            - Compare patterns across different experiments.
            - Look for concentrations of high-relevance documents (green) at top ranks.
            - Identify queries with many unjudged documents (gray).
            - Use the depth slider to focus on specific ranking ranges.
            """
            )

    if "qmed_selected_runs" in st.session_state:
        # Slider for selecting ranking depth
        st.session_state.ranking_depth = st.slider(
            "Select Ranking Depth", min_value=5, max_value=100, value=25, step=5
        )

        plot_rankings_docs_rel_ids(
            st.session_state.qmed_selected_qrels,
            st.session_state.qmed_selected_runs,
            st.session_state.ranking_depth,
        )
    else:
        st.warning("No runs selected. Please select runs to visualize the rankings.")

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
    """<h4 style="text-align:center;"><span style="color:red;">To export the report as PDF press (⌘+P or Ctrl+P)</span></h4>""",
    unsafe_allow_html=True,
)
