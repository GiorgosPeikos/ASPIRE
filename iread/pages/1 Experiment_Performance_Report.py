import streamlit as st
import os
import math
import pandas as pd
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
from utils.data_handler import load_run_data, load_qrel_data, load_query_data
from utils.ui import load_css
from utils.eval_multiple_exp import evaluate_multiple_runs_custom, get_doc_intersection, get_docs_retrieved_by_all_systems
from utils.eval_core import evaluate_single_run, return_available_measures, get_relevant_and_unjudged, generate_prec_recall_graphs
from utils.plots import plot_dist_of_retrieved_docs, plot_precision_recall_curve


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
            f"""<div style="text-align: center;">Evaluating the <span style="color:red;">{", ".join(selected_runs_files).replace('.txt', '').replace('.csv', '')}</span> experiments using the <span style="color:red;">{qrels_file}</span> qrels.</div>""",
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

    if 'me_selected_runs' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")
    else:
        if st.session_state.me_max_relevance >= 2:
            st.session_state.me_relevance_threshold = st.slider(
                "Select from the Available Relevance Thresholds (Slide)",
                min_value=1,
                max_value=2,
                value=1,
                key="me_slider3",
            )

            if 'me_prev_relevance_threshold' not in st.session_state:
                st.session_state.me_prev_relevance_threshold = 1

            if st.session_state.me_relevance_threshold != st.session_state.me_prev_relevance_threshold:
                st.session_state.me_prev_relevance_threshold = st.session_state.me_relevance_threshold
        else:
            st.session_state.me_relevance_threshold = 1
            st.write("""**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""", unsafe_allow_html=True)

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
                    overall_measures_results = {}
                    for measure_name in overall_measures:
                        overall_measures_results[measure_name] = evaluate_single_run(
                            st.session_state.me_selected_qrels, st.session_state.me_selected_runs[run_file], measure_name, st.session_state.me_relevance_threshold
                        )

                    df_measures = pd.DataFrame([overall_measures_results],
                                               index=[run_file.replace('.txt', '').replace('.csv', '')])

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

                    df_prec_measures = pd.DataFrame(precision_measures_results, index=[run_file.replace('.txt', '').replace('.csv', '')]).transpose()

                    # Add precision measures to the combined DataFrame
                    precision_measures_combined[run_file.replace('.txt', '').replace('.csv', '')] = df_prec_measures.iloc[:, 0]

                    # Calculate and add recall measures to the combined DataFrame
                    recall_measures_results = {
                        "Recall@50": evaluate_single_run(
                            st.session_state.me_selected_qrels, st.session_state.me_selected_runs[run_file], "R@50", st.session_state.me_relevance_threshold
                        ),
                        "Recall@1000": evaluate_single_run(
                            st.session_state.me_selected_qrels, st.session_state.me_selected_runs[run_file], "R@1000", st.session_state.me_relevance_threshold
                        )
                    }
                    df_recall_measures = pd.DataFrame([recall_measures_results], index=[run_file.replace('.txt', '').replace('.csv', '')])
                    recall_measures_combined = pd.concat([recall_measures_combined, df_recall_measures.transpose()], axis=1)

            # Clear the progress bar and text
            progress_bar.empty()
            progress_text.empty()

            # Display the combined DataFrames
            st.markdown("<h5>Overall Measures Combined</h5>", unsafe_allow_html=True)
            st.dataframe(overall_measures_combined, use_container_width=True)

            st.markdown("<h5>Precision Measures Combined</h5>", unsafe_allow_html=True)
            st.dataframe(precision_measures_combined, use_container_width=True)

            st.markdown("<h5>Recall Measures Combined</h5>", unsafe_allow_html=True)
            st.dataframe(recall_measures_combined, use_container_width=True)

st.divider()

# Common Evaluation Measures
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Experimental Evaluation</span></h3>""", unsafe_allow_html=True)
    _, _, custom_user, default_measures, _, _ = return_available_measures()

    if 'me_selected_qrels' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")
    else:

        # Get the list of selected run files
        selected_runs_files = list(st.session_state.me_selected_runs.keys())

        if 'me_selected_qrels' in st.session_state and not st.session_state.me_selected_qrels.empty:
            # Define possible values for correction
            correction_methods = ['Bonferroni', 'Holm', 'Holm-Sidak']

            if 'load' not in st.session_state:
                st.session_state.baseline = list(st.session_state.me_selected_runs.keys())[0]
                st.session_state.selected_correction_alpha = 0.05
                st.session_state.selected_correction = correction_methods[0]

                st.markdown(
                    f"""Statistical significance is tested against the selected baseline 
                    (<span style="color:red;">{st.session_state.baseline}</span>) 
                    using a paired two-sided t-test at a significance level 
                    (<span style="color:red;">{st.session_state.selected_correction_alpha}</span>).
                    Multiple testing correction is performed using the (<span style="color:red;">{st.session_state.selected_correction}</span>) method.
                    """,
                    unsafe_allow_html=True
                )

            # Just a space
            st.write("#")

            # Create columns
            col1, col2 = st.columns([3, 2])  # Adjust the column width ratio as needed

            # Just a space
            st.write("#")

            with col1:
                if st.session_state.me_max_relevance >= 2:
                    st.session_state.me_relevance_threshold = st.slider(
                        "Select from the Available Relevance Thresholds (Slide)",
                        min_value=1,
                        max_value=2,
                        value=1,
                        key="me_slider4",
                    )

                    if 'me_prev_relevance_threshold' not in st.session_state:
                        st.session_state.me_prev_relevance_threshold = 1

                    if st.session_state.me_relevance_threshold != st.session_state.me_prev_relevance_threshold:
                        st.session_state.me_prev_relevance_threshold = st.session_state.me_relevance_threshold
                else:
                    st.session_state.me_relevance_threshold = 1
                    st.write("""**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""", unsafe_allow_html=True)

                st.session_state.selected_correction_alpha = st.slider(
                    "Correction Value (alpha)",
                    min_value=0.01,
                    max_value=0.05,
                    value=0.05,
                    step=0.04,
                    key="me_slider_alpha",
                )

            with col2:
                # Single select component for correction method with bonferroni selected by default
                st.session_state.selected_correction = st.selectbox(
                    "Select a correction method:",
                    correction_methods,
                    index=0,
                )
                if selected_runs_files:
                    # Add a selectbox to choose the baseline run
                    st.session_state.baseline = st.selectbox(
                        "Select a baseline run file:",
                        list(st.session_state.me_selected_runs.keys()),
                    )

            st.session_state.results_table, style_table = evaluate_multiple_runs_custom(st.session_state.me_selected_qrels, st.session_state.me_selected_runs, default_measures,
                                                                                        st.session_state.me_relevance_threshold,
                                                                                        st.session_state.baseline, None, st.session_state.selected_correction,
                                                                                        st.session_state.selected_correction_alpha)

            if 'load' not in st.session_state:
                st.session_state.load = True
            else:
                st.markdown(
                    f"""Statistical significance is tested against the selected baseline 
                    (<span style="color:red;">{st.session_state.baseline}</span>) 
                    using a paired two-sided t-test at a significance level 
                    (<span style="color:red;">{st.session_state.selected_correction_alpha}</span>).
                    Multiple testing correction is performed using the (<span style="color:red;">{st.session_state.selected_correction}</span>) method.
                    """,
                    unsafe_allow_html=True
                )

            if not st.session_state.results_table.empty:
                # Display the table in Streamlit
                st.dataframe(st.session_state.results_table.style.apply(lambda _: style_table, axis=None), use_container_width=True)

                # Add a legend
                st.markdown("""
                Format is <span style="color:red;">Measure | <sup>p-value</sup> <sub>corrected p-value</sub></span>. If the observed difference from the baseline is statistically significant, 
                the background of the measure is green. The highest value per measure is underscored.
                """, unsafe_allow_html=True)

            st.divider()

            # Splitting the container
            col = st.columns(2)

            # Initialize session state variables if they don't exist
            if 'selected_measures' not in st.session_state:
                st.session_state.selected_measures = custom_user[0:1]  # Default selected measures
            if 'selected_cutoff' not in st.session_state:
                st.session_state.selected_cutoff = 10  # Default cutoff value

            with col[0]:
                selected_measures = st.multiselect("Select additional measures:", custom_user, default=custom_user[5:6])
            with col[1]:
                selected_cutoff = st.number_input("Enter measure cutoff value:", min_value=1, value=25, max_value=1000, step=1)

            # Update session state with current selections
            st.session_state.selected_measures = selected_measures
            st.session_state.selected_cutoff = selected_cutoff

            st.session_state.results_table_custom, style_table_custom = evaluate_multiple_runs_custom(st.session_state.me_selected_qrels, st.session_state.me_selected_runs, selected_measures,
                                                                                                      st.session_state.me_relevance_threshold, st.session_state.baseline,
                                                                                                      st.session_state.selected_cutoff,
                                                                                                      st.session_state.selected_correction, st.session_state.selected_correction_alpha)

            if not st.session_state.results_table_custom.empty:
                # Display the table in Streamlit
                st.dataframe(st.session_state.results_table_custom.style.apply(lambda _: style_table_custom, axis=None), use_container_width=True)

                # Add a legend
                st.markdown("""
                Format is <span style="color:red;">Measure | <sup>p-value</sup> <sub>corrected p-value</sub></span>. If the observed difference from the baseline is statistically significant, 
                the background of the measure is green. The highest value per measure is underscored.
                """, unsafe_allow_html=True)

st.divider()

# Positional Distribution of Relevant Retrieved Documents
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Positional Distribution of Relevant and Unjudged Retrieved Documents</span></h3>""", unsafe_allow_html=True)

    if 'me_selected_qrels' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")

    else:
        if 'me_selected_runs' in st.session_state:
            num_runs = len(st.session_state.me_selected_runs)
            num_rows = math.ceil(num_runs / 2)

            for i in range(num_rows):
                cols = st.columns(2)
                for j in range(2):
                    run_index = i * 2 + j
                    if run_index < num_runs:
                        run_key, run_value = list(st.session_state.me_selected_runs.items())[run_index]
                        ranking_per_relevance = get_relevant_and_unjudged(st.session_state.get('me_selected_qrels'), run_value)

                        with cols[j]:
                            st.markdown(f"""#### Experiment: <span style="color:red;"> {str(run_key).replace('.txt', '').replace('.csv', '')}</span>""", unsafe_allow_html=True)
                            plot_dist_of_retrieved_docs(ranking_per_relevance)
st.divider()


# Precision/Recall Curve
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Precision/Recall Curve</span></h3>""", unsafe_allow_html=True)

    if 'me_selected_qrels' not in st.session_state:
        st.warning("Please select retrieval experiment and qrels to begin your evaluation.", icon="⚠")
    else:
        if st.session_state.me_max_relevance >= 2:
            st.session_state.me_relevance_threshold = st.slider(
                "Select from the Available Relevance Thresholds (Slide)",
                min_value=1,
                max_value=2,
                value=1,
                key="me_slider5",
            )

            if 'me_prev_relevance_threshold' not in st.session_state:
                st.session_state.me_prev_relevance_threshold = 1

            if st.session_state.me_relevance_threshold != st.session_state.me_prev_relevance_threshold:
                st.session_state.me_prev_relevance_threshold = st.session_state.me_relevance_threshold
        else:
            st.session_state.me_relevance_threshold = 1
            st.write("""**Relevance judgements are binary, so <span style="color:red;">relevance threshold is set to 1.</span>**""", unsafe_allow_html=True)

        if 'me_selected_runs' in st.session_state:
            num_runs = len(st.session_state.me_selected_runs)
            num_rows = math.ceil(num_runs / 2)

            for i in range(num_rows):
                cols = st.columns(2)
                for j in range(2):
                    run_index = i * 2 + j
                    if run_index < num_runs:
                        run_key, run_value = list(st.session_state.me_selected_runs.items())[run_index]

                        with cols[j]:
                            st.markdown(f"""#### Experiment: <span style="color:red;"> {str(run_key).replace('.txt', '').replace('.csv', '')}</span>""", unsafe_allow_html=True)

                            # Generate and plot the precision-recall curve for this run
                            prec_recall_graph = generate_prec_recall_graphs(st.session_state.me_relevance_threshold, st.session_state.me_selected_qrels, run_value)
                            plot_precision_recall_curve(prec_recall_graph, st.session_state.me_relevance_threshold)

st.divider()


# Retrieved Document Intersection
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Retrieved Document Intersection</span></h3>""", unsafe_allow_html=True)

    if 'me_selected_runs' not in st.session_state or len(st.session_state.me_selected_runs) < 2:
        st.warning("This analysis requires at least two retrieval experiments to be selected.", icon="⚠")

    else:
        st.write("""
             This analysis calculates the intersection of retrieved documents between a selected baseline experiment and other retrieval experiments.
             It showcases the similarity between the document rankings across different retrieval approaches.
             """)

        col1, col2 = st.columns(2)

        with col1:
            st.session_state.baseline = st.selectbox(
                "Select a baseline run file:",
                list(st.session_state.me_selected_runs.keys()),
                key="selectbox1",
            )

        with col2:
            selected_cutoff_inter = st.number_input("Top-ranked documents considered (ranking cutoff value):", min_value=1, value=10, max_value=1000, step=1, key='cutoff2')

        st.write(f"""
               **Selected Baseline**:  <span style="color:red;">{str(st.session_state.baseline).replace('.txt','').replace('.csv','')}</span>. All other experiments will be compared against this 
               baseline.

               **Cutoff Value**: <span style="color:red;">{selected_cutoff_inter}</span>. The cutoff value determines how many top-ranked documents from each experiment are considered in the intersection analysis.
               """, unsafe_allow_html=True)

        intersection_results = get_doc_intersection(st.session_state.me_selected_runs, st.session_state.baseline, selected_cutoff_inter)

        st.dataframe(intersection_results)

        with col1:
            st.write(f"""
                <span style="color:red;">**Interpretation**:</span>
                
                - Intersected Documents: The number of documents that appear in both the baseline and the compared experiments within the top <span style="color:red;">{selected_cutoff_inter}</span> results.
                
                - Total Documents: The total number of documents considered (number of queries × cutoff value).
                
                - Intersection Percentage: The percentage of documents that intersect, calculated as (Intersected Documents / Total Documents) × 100.
        
                <span style="color:red;">A higher intersection percentage indicates greater similarity with the baseline results.</span>
                """, unsafe_allow_html=True)

        with col2:
            st.write("""
               <span style="color:red;">**Potential Next Steps**:</span>
               - Analyze experiments with high intersection percentages to understand what makes them similar to the baseline.
               - Investigate experiments with low intersection percentages to determine if they're introducing beneficial diversity or are potentially underperforming.
               - Conduct a query-level analysis to identify which types of queries lead to high or low intersection across experiments.
               - Consider combining systems with low intersection to potentially improve overall performance.
               """, unsafe_allow_html=True)

st.divider()

# Frequently Retrieved Documents
with st.container():
    st.markdown("""<h3>Retrieval Performance - <span style="color:red;">Documents Retrieved by All Systems</span></h3>""", unsafe_allow_html=True)

    if 'me_selected_runs' not in st.session_state or len(st.session_state.me_selected_runs) < 2:
        st.warning("This analysis requires at least two retrieval experiments to be selected.", icon="⚠")
    else:
        st.write("""
        This analysis identifies documents that are retrieved by all selected retrieval systems within a specified cutoff rank.
        These documents represent a consensus among different retrieval approaches and may be particularly relevant or central to the queries.
        """)

        col1, col2 = st.columns(2)

        with col1:
            selected_cutoff = st.number_input(
                "Number of top-ranked documents considered:",
                min_value=1,
                value=1,
                max_value=1000,
                step=1,
                key='cutoff_retrieved_docs',
            )
            st.write(f"""
            **Cutoff Value**: {selected_cutoff}. The analysis considers the top {selected_cutoff} ranked documents for each query across all experiments.
            """)

        with col2:
            sample_documents = st.number_input(
                "Documents (IDs) retrieved by all systems:",
                min_value=1,
                value=10,
                max_value=100,
                step=1,
                key='sampled_retrieved_docs',
            )
            st.write(f"""
            **Sample Size**: {sample_documents}. The number of document IDs to display in the results.
            """)

        retrieved_docs, total_queries, query_ids = get_docs_retrieved_by_all_systems(st.session_state.me_selected_runs, selected_cutoff, sample_documents)

        st.write(f"**Total Queries with documents retrieved by all systems: {total_queries}**")
        st.write("Query IDs:", ", ".join(map(str, query_ids)))
        # st.write(", ".join(map(str, query_ids)))

        st.write(f"\n**Sample of documents retrieved by all {len(st.session_state.me_selected_runs)} systems:**")

        if retrieved_docs:
            st.write(", ".join(map(str, retrieved_docs)))

            with col1:
                st.write("""
                       <span style="color:red;">**Interpretation**:</span>
                       
                       - The query IDs listed above represent queries where at least one document was retrieved by all systems within the specified cutoff.
                       - For instance, if the cutoff value is 1, the analysis presents those queries for which all systems retrieve the same document in the 1st rank position.
                       - The documents listed are a sample of those retrieved by all systems, prioritized by frequency across queries. Sample size can be adjusted.
                       - These documents may be highly relevant to multiple queries or represent core content in your collection.
                       - Their consistent retrieval across all systems suggests they are important in the context of your retrieval task.""", unsafe_allow_html=True)
            with col2:
                st.write("""
                        <span style="color:red;">**Potential Next Steps**:</span>
                        
                       - Examine the listed queries to understand what types of information needs lead to consistent retrieval across systems.
                       - Analyze the sample documents to identify characteristics that make them consistently retrievable across the identified queries.
                       - If the number of queries or documents is lower than expected, consider increasing the cutoff value or investigating differences in retrieval approaches.
                       - Use these results as a starting point for in-depth relevance assessment or to refine your retrieval approaches.
                       """, unsafe_allow_html=True)
        else:
            st.write(f"No documents were retrieved by all systems in the top {selected_cutoff} results.")

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
