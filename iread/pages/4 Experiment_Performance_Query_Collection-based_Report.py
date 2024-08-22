import streamlit as st
import os
import numpy as np
st.set_page_config(layout="wide", initial_sidebar_state="collapsed", )
from utils.data_handler import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.eval_query_collection import analyze_query_judgements, find_multi_query_docs, find_ranked_pos_of_multi_query_docs, display_further_details_multi_query_docs
from utils.plots import plot_query_relevance_judgements, plot_multi_query_docs


# Load custom CSS
load_css("css/styles.css")

# Title for the page
st.markdown("""<div style="text-align: center;"><h1>Retrieval Evaluation Report based on the Retrieved Documents<h1></div>""", unsafe_allow_html=True)

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
if st.button("Begin the Experimental Evaluation!", key='qmed_stButtonCenter'):

    if not qrels_file:
        st.write("Please select Qrels file to proceed.")
        st.stop()

    if not selected_runs_files:
        st.write("Please select at least one Retrieval Run file to proceed.")
        st.stop()

    # Load Qrels data
    st.session_state.qmed_selected_qrels = load_qrel_data(os.path.join(qrels_dir, qrels_file))
    st.session_state.qmed_max_relevance = st.session_state.qmed_selected_qrels["relevance"].max()

    # Load selected runs data
    st.session_state.qmed_selected_runs = {}
    for run_file in selected_runs_files:
        st.session_state.qmed_selected_runs[run_file] = load_run_data(os.path.join(runs_dir, run_file))

    if st.session_state.qmed_selected_runs:
        st.markdown(
            f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{", ".join(selected_runs_files).replace('.txt', '').replace('.csv', '')}</span> experiments using the <span style="color:red;">{qrels_file}</span> qrels.</div>""",
            unsafe_allow_html=True)

    if queries_file:
        st.session_state.qmed_selected_queries = load_query_data(os.path.join(queries_dir, queries_file))
        st.markdown(
            f"""<div style="text-align: center;">This experiment is associated with the <span style="color:red;">{queries_file}</span> queries.</div>""",
            unsafe_allow_html=True)

st.divider()

# Functionality that allows to randomly select queries for analysis, size and queries.
if 'qmed_selected_queries' in st.session_state and not st.session_state.qmed_selected_queries.empty:
    if len(st.session_state.qmed_selected_queries) > 251:
        with st.container():
            st.write("""<h3>Query Sampling - <span style="color:red;">Sampling Queries to Facilitate Experiment Analysis</span></h3>""", unsafe_allow_html=True)

            # Initialize random_state in session state if it doesn't exist
            if 'qmed_random_state' not in st.session_state:
                st.session_state.qmed_random_state = 42
            if 'qmed_random_size' not in st.session_state:
                st.session_state.qmed_random_size = 250

            st.write(f"""<div style="text-align: center;">  ⚠️ Note: Too many available queries (<span style="color:red;">{len(st.session_state.qmed_selected_queries)}</span>).
            To enhance the following analysis, a random set will be used. Please select the following:</div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.session_state.qmed_random_size = st.number_input(
                    "Set Number of queries to be randomly selected (default: 49)",
                    min_value=1,
                    value=st.session_state.qmed_random_size,
                    max_value=len(st.session_state.qmed_selected_queries),
                    step=1
                )
            with col2:
                st.session_state.qmed_random_state = st.number_input(
                    "Set random state for query selection (default: 42)",
                    min_value=1,
                    value=st.session_state.qmed_random_state,
                    max_value=100,
                    step=1
                )

            # Main content logic
            st.write(f"""<div style="text-align: center;"> A total of <span style="color:red;">{st.session_state.qmed_random_size}</span> random queries have been
            selected based on a random state equal to <span style="color:red;">{st.session_state.qmed_random_state}</span> and will be used for the upcoming analyses.</div>""", unsafe_allow_html=True)

            st.session_state.qmed_selected_queries_random = st.session_state.qmed_selected_queries.sample(n=st.session_state.qmed_random_size, random_state=st.session_state.qmed_random_state)

            st.write(f"""<div style="text-align: center;"> Number of randomly selected queries that would be used for analysis: <span style="color:red;"
                >{len(st.session_state.qmed_selected_queries_random)}</span></div>""", unsafe_allow_html=True)

            query_ids = np.sort(st.session_state.qmed_selected_queries_random.query_id.values)
            query_ids_str = ", ".join(map(str, query_ids))
            st.write(f"Selected Query (IDs): {query_ids_str}")

            st.divider()
    else:
        if 'qmed_selected_queries_random' not in st.session_state:
            st.session_state.qmed_selected_queries_random = st.session_state.qmed_selected_queries
            st.write(f"""<div style="text-align: center;"> All <span style="color:red;">{len(st.session_state.qmed_selected_queries_random)}</span> provided queries will be used for the 
            following analyses.</div>""", unsafe_allow_html=True)

            st.divider()


# Per query Relevance Judgements
with st.container():
    st.markdown("""<h3>Document Collection - <span style="color:red;">Relevance Judgments per Query</span></h3>""", unsafe_allow_html=True)

    with st.expander("See Analysis Details and Interpretations"):
        st.write("<center><b>The analysis leverages only the provided Qrels file and a subset of queries in case the available queries are more than 250. </b></center>", unsafe_allow_html=True)

    if 'qmed_selected_runs' not in st.session_state:
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
        st.session_state.qmed_query_rel_judg = plot_query_relevance_judgements(st.session_state.qmed_selected_qrels)
        analysis_results = analyze_query_judgements(st.session_state.qmed_query_rel_judg)

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
            st.dataframe(st.session_state.qmed_selected_queries_random[['query_id', 'query_text']], use_container_width=True, hide_index=True)

st.divider()


# Documents that have relevance assessment for more than one query
with st.container():
    st.markdown("""<h3>Document Collection - <span style="color:red;">Documents with Relevance Judgments for Multiple Queries</span></h3>""", unsafe_allow_html=True)

    with st.expander("See Analysis Details and Interpretations"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Overview")
            st.write("""
            This analysis identifies and examines documents that have received relevance assessments for more than one query. These documents are particularly important because they help reveal 
            patterns in how different documents are assessed w.r.t to different queries.
            """)

        with col2:
            st.markdown("##### Potential Usages")
            st.write("""
            - **Document Relevance Across Queries:** It highlights how the same document might be relevant to multiple queries, indicating the versatility or generality of the document's content.
            - **Query Overlap and Distinction:** Analyzing shared documents between queries helps understand the overlap or distinction in query intent and potential retrieval performance.
            """)

        st.write("<center><b>The analysis leverages only the provided Qrels file and is based on all of the available Queries in the collection!</b></center>", unsafe_allow_html=True)

    if 'qmed_selected_qrels' in st.session_state:
        multi_query_docs = find_multi_query_docs(st.session_state.qmed_selected_qrels)

        if not multi_query_docs.empty:
            st.write(f"There are <span style='color:red;'>{len(multi_query_docs)}</span> documents whose relevance have been assessed w.r.t. multiple queries.", unsafe_allow_html=True)

            st.session_state.qmed_num_docs = st.slider("Number of documents to display in the graph", min_value=1, max_value=len(multi_query_docs), value=min(100, len(multi_query_docs)))

            # Show the plot
            fig = plot_multi_query_docs(multi_query_docs.head(st.session_state.qmed_num_docs))
            st.plotly_chart(fig, use_container_width=True)

            search_doc = st.text_input("Search for a specific document ID")
            if search_doc:
                filtered_docs = multi_query_docs[multi_query_docs.doc_id.str.contains(str(search_doc), case=False)]
                if not filtered_docs.empty:
                    st.dataframe(filtered_docs, hide_index=True)
                else:
                    st.write("No matching documents found.")

            with st.expander("Further Details"):
                display_further_details_multi_query_docs(multi_query_docs, st.session_state.qmed_num_docs)

            with st.expander("What was the ranking position and relevance judgment of these documents across experiments and queries?"):
                find_ranked_pos_of_multi_query_docs(multi_query_docs, st.session_state.qmed_num_docs, st.session_state.qmed_selected_runs, st.session_state.qmed_selected_qrels)

        else:
            st.write("No documents found with relevance assessments for multiple queries.")
    else:
        st.warning("Please load qrels data to perform this analysis.", icon="⚠")

st.divider()


# Documents that have been retrieved by all systems.
with st.container():
    st.markdown("""<h3>Document Collection - <span style="color:red;">Documents retrieved in all Experiments</span></h3>""", unsafe_allow_html=True)

    with st.expander("See Analysis Details and Interpretations"):
        col1, col2 = st.columns(2)

        st.write('details')

    if 'qmed_selected_runs' in st.session_state:
        st.write()

st.divider()


# Documents that have been retrieved by 1,2,3,5 systems.
with st.container():
    st.markdown("""<h3>Document Collection - <span style="color:red;">Documents retrieved in a few Experiments</span></h3>""", unsafe_allow_html=True)

    with st.expander("See Analysis Details and Interpretations"):
        col1, col2 = st.columns(2)

        st.write('details')

    if 'qmed_selected_runs' in st.session_state:
        st.write()

st.divider()


# The box like plot for the top 10 positions showing how the documents are ranked.
with st.container():
    st.markdown("""<h3>Document Collection - <span style="color:red;">Retrieved Documents, Relevance, Ranking Position</span></h3>""", unsafe_allow_html=True)

    with st.expander("See Analysis Details and Interpretations"):
        col1, col2 = st.columns(2)

        st.write('details')

    if 'qmed_selected_runs' in st.session_state:
        st.write()

st.divider()

