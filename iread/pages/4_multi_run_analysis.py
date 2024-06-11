import streamlit as st
import pandas as pd


from utils.analysis_styling import color_max_min_column, color_max_min_row
from utils.data import print_session_data, load_qrel_data, load_run_data
from utils.evaluation_measures import initialize_results, return_available_measures, evaluate_single_run
from utils.ui import query_selector, multi_run_selector

print_session_data()
query_selector()
multi_run_selector()

st.title("This page contains tha content of comparing two different retrieval runs.")

run_paths = [st.session_state[x] for x in st.session_state if x.startswith("run__")]
run_names = [run.split('/')[-1] for run in run_paths]

if "selected_qrels" not in st.session_state and not run_paths:
    st.error("Errors in Calculations. No run selected.")
    st.stop()

qrel_path = st.session_state["selected_qrels"]

runs = [load_run_data(run_path) for run_path in run_paths]
qrels = load_qrel_data(qrel_path)
relevance_max = qrels["relevance"].max()  # Get maximum relevance level

#
# filename_with_suffix = os.path.basename(run_path)
# filename_without_suffix = os.path.splitext(filename_with_suffix)[0]
# st.markdown(
#     f"""<div style="text-align: center;"><h1><u>  Evaluating the <span style="color:red;"> <u>{filename_without_suffix} </u></span> Experiment </u> </h1></div>""",
#     unsafe_allow_html=True,
# )

st.header("Mean Performance Evaluation")

st.sidebar.subheader("Additional settings")
# Slider for relevance threshold
relevance_threshold = st.sidebar.slider(
    "Select from the Available Relevance Thresholds (Slide)",
    min_value=1,
    max_value=int(relevance_max),
    value=1,
)
if "prev_relevance_threshold" not in st.session_state:
    st.session_state.prev_relevance_threshold = relevance_threshold
if relevance_threshold != st.session_state.prev_relevance_threshold:
    st.session_state["results_standard"] = {}
    st.session_state["results_extra"] = {}
    st.session_state.prev_relevance_threshold = relevance_threshold

freq_measures, rest_measures, custom_user = return_available_measures()

initialize_results()
measures_results = {run_name: {} for run_name in run_names}


freq_measures_results = {}

baseline_measures_results = {}
for measure_name in freq_measures:
    for run_name, run in zip(run_names, runs):
        measures_results[run_name][measure_name] = evaluate_single_run(qrels, run, measure_name, relevance_threshold
                                                                  )
df = pd.DataFrame.from_dict(measures_results).T

if st.button("transpose?"):
    st.dataframe(df.T.style.apply(color_max_min_row, axis=None))
else:
    st.dataframe(df.style.apply(color_max_min_column, axis=None))

if st.button("to latex"):
    st.write(df.to_latex())
