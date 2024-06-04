import streamlit as st
import pandas as pd
import os


def read_data(folder_path: str):
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    data_frames = {file: pd.read_csv(os.path.join(folder_path, file)) for file in files}
    return data_frames


# Function to load run data with caching to optimize performance
@st.cache_data
def load_run_data(run_path):
    # Reads a CSV file into a DataFrame with specified columns and delimiters
    return pd.read_csv(
        run_path,
        names=["query_id", "iteration", "doc_id", "rank", "score", "tag"],
        delimiter="[ \t]",
        dtype={
            "query_id": str,
            "iteration": str,
            "doc_id": str,
            "rank": str,
            "score": float,
            "tag": str,
        },
        index_col=None,
        engine="python",
        header=0,
    )


# Function to load qrel data with caching
@st.cache_data
def load_qrel_data(qrel_path):
    # Reads a CSV file into a DataFrame for qrel data
    return pd.read_csv(
        qrel_path,
        names=["query_id", "iteration", "doc_id", "relevance"],
        delimiter="[ \t]",
        dtype={"query_id": str, "iteration": str, "doc_id": str, "relevance": int},
        index_col=None,
        engine="python",
        header=0,
    )


def load_css(file_name: str):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def print_session_data() -> None:
    if "qrels" in st.session_state:
        st.write(f"Loaded qrels: {st.session_state['qrels']}")
    else:
        st.write("No qrels available")

    if "runs" in st.session_state:
        st.write(f"Current runs: {st.session_state['runs']}")
    else:
        st.write("No runs available")
