import streamlit as st
import os
st.set_page_config(layout="wide", initial_sidebar_state="collapsed", )
from utils.data_handler import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.eval_query_collection import analyze_query_judgements
from utils.plots import plot_query_relevance_judgements


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

# Per query Relevance Judgements
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Relevance Judgments per Query</span></h3>""", unsafe_allow_html=True)

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














# Documents that have relevance assessment for more than one query.

# Documents that have been retrieved by all systems.

# Documents that have been retrieved by 1,2,3,5 systems.

# The box like plot for the top 10 positions showing how the documents are ranked.
