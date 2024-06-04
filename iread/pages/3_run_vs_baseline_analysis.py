import streamlit as st

from utils.data import print_session_data
from utils.ui import query_selector, single_run_selector

print_session_data()

query_selector()
single_run_selector(title="Baseline run", session_key="baseline_run")
single_run_selector()

if "selected_run" not in st.session_state:
    st.error("Errors in Calculations. No run selected.")
    st.stop()
