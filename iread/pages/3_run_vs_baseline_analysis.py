import os

import pandas as pd
import streamlit as st

from utils.data import print_session_data, load_run_data, load_qrel_data
from utils.evaluation_measures import (
    return_available_measures,
    initialize_results,
    evaluate_single_run,
)
from utils.ui import query_selector, single_run_selector

print_session_data()

query_selector()
single_run_selector(title="Baseline run", session_key="baseline_run")
single_run_selector()

if not any(
    bad_word in st.session_state
    for bad_word in ["selected_run", "selected_qrels", "baseline_run"]
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
    index=["BASELINE", "filename_without_suffix"],
)


def color_max_min_column(x):
    top_colour = "background-color: #b7f5ae"
    low_colour = "background-color: #f78c81"
    top_score = x.eq(x.max())
    low_score = x.eq(x.min())

    df1 = pd.DataFrame("", index=x.index, columns=x.columns)
    return df1.mask(top_score, top_colour).mask(low_score, low_colour)


st.dataframe(df.style.apply(color_max_min_column, axis=None))
