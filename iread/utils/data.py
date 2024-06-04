import streamlit as st
import pandas as pd
import os
import xml.etree.ElementTree as ET


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


@st.cache_data
def load_query_data(query_path):

    # Determine the file extension
    file_extension = os.path.splitext(query_path)[1].lower()

    # Process based on file type
    if file_extension in [".csv", ".txt"]:
        return pd.read_csv(
            query_path,
            names=["query_id", "query_text"],
            delimiter="[ \t]",
            dtype={"query_id": str, "iteration": str, "doc_id": str, "relevance": int},
            index_col=None,
            engine="python",
            header=0,
        )
    elif file_extension == ".xml":
        # Parse XML and extract the required data
        tree = ET.parse(query_path)
        root = tree.getroot()

        # Create lists to store query_id and query_text
        query_ids = []
        query_texts = []

        # Iterate through each 'topic' in the XML and extract required information
        for topic in root.findall(".//topic"):
            query_id = topic.get("number")
            query_text = "".join(topic.itertext()).strip()
            query_ids.append(query_id)
            query_texts.append(query_text)

        # Create a DataFrame from the lists
        return pd.DataFrame({"query_id": query_ids, "query_text": query_texts})
    else:
        raise ValueError("Unsupported file format")


def print_session_data() -> None:
    """Only for debug to make sure that we loaded correct data."""
    if "qrels" in st.session_state:
        st.write(f"Loaded qrels: {st.session_state['qrels']}")
    else:
        st.write("No qrels available")

    if "runs" in st.session_state:
        st.write(f"Current runs: {st.session_state['runs']}")
    else:
        st.write("No runs available")
