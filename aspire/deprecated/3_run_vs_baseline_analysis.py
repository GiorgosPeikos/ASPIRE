import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from deprecated.analysis_styling import color_max_min_column, color_max_min_row
from utils.data_handler import (load_qrel_data, load_run_data,
                                print_session_data)
from utils.eval_core import (evaluate_single_run, initialize_results,
                             return_available_measures)
from utils.eval_single_exp import find_unjudged
from utils.ui import query_selector, single_run_selector

print_session_data()

query_selector()
single_run_selector(title="Baseline run", session_key="baseline_run")
single_run_selector()

if not any(
    item in st.session_state
    for item in ["selected_run", "selected_qrels", "baseline_run"]
):
    st.error("Errors in Calculations. No run selected.")
    st.stop()

run_path = st.session_state["selected_run"]
baseline_path = st.session_state["baseline_run"]
qrel_path = st.session_state["selected_qrels"]
run = load_run_data(run_path)
baseline_run = load_run_data(baseline_path)
qrels = load_qrel_data(qrel_path)
relevance_max = qrels["relevance"].max()  # Get maximum relevance level

# Displaying the experiment's name
filename_with_suffix = os.path.basename(run_path)
filename_without_suffix = os.path.splitext(filename_with_suffix)[0]
st.markdown(
    f"""<div style="text-align: center;"><h1><u>  Evaluating the <span style="color:red;"> <u>{filename_without_suffix} </u></span> Experiment </u> </h1></div>""",
    unsafe_allow_html=True,
)

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
freq_measures_results = {}
baseline_measures_results = {}
for measure_name in freq_measures:
    freq_measures_results[measure_name] = evaluate_single_run(
        qrels, run, measure_name, relevance_threshold
    )
    baseline_measures_results[measure_name] = evaluate_single_run(
        qrels, baseline_run, measure_name, relevance_threshold
    )

df = pd.DataFrame(
    [baseline_measures_results, freq_measures_results],
    index=["BASELINE", filename_without_suffix],
)

if st.button("transpose?"):
    st.dataframe(df.T.style.apply(color_max_min_row, axis=None))
else:
    st.dataframe(df.style.apply(color_max_min_column, axis=None))

st.write("Number of unjudged documents depending on the cutoff threshold.")
cutoffs = st.slider(
    label="Select cutoffs", min_value=10, max_value=1000, value=(10, 100), step=10
)
means_baseline = []
means_run = []
for cutoff in cutoffs:
    unjudged_run = find_unjudged(run=run, qrels=qrels, cutoff=cutoff)
    unjudged_baseline = find_unjudged(run=baseline_run, qrels=qrels, cutoff=cutoff)
    means_run.append(len(unjudged_run) / len(qrels["query_id"].unique()))
    means_baseline.append(len(unjudged_baseline) / len(qrels["query_id"].unique()))

df_means = pd.DataFrame(
    {"Cutoffs": cutoffs, "Baseline": means_baseline, "Run": means_run}
)

plt.figure(figsize=(10, 6))
plt.plot(
    df_means["Cutoffs"], df_means["Baseline"], marker="", color="blue", label="Baseline"
)
plt.plot(df_means["Cutoffs"], df_means["Run"], marker="", color="red", label="Run")
plt.legend()

st.pyplot(plt)
