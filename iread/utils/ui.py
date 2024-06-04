import streamlit as st
import os


def query_selector():
    st.subheader("Qrels Files")
    qrels_dir = os.path.join(
        os.path.dirname(os.getcwd()), "retrieval_experiments/qrels"
    )
    if os.path.exists(qrels_dir):
        selected_qrels = st.selectbox("", os.listdir(qrels_dir))
        st.session_state["selected_qrels"] = os.path.join(qrels_dir, selected_qrels)


def single_run_selector():
    st.subheader("Retrieval Experiments")
    experiments_dir = os.path.join(
        os.path.dirname(os.getcwd()), "retrieval_experiments/retrieval_runs"
    )
    if os.path.exists(experiments_dir):
        selected_run = st.selectbox("", os.listdir(experiments_dir))
        st.session_state["selected_run"] = os.path.join(experiments_dir, selected_run)


def load_css(file_name: str):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
