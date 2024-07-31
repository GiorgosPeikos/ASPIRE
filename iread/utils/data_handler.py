import os
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import streamlit as st
from typing import Union, List
from pathlib import Path


def read_data(folder_path: str):
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    data_frames = {file: pd.read_csv(os.path.join(folder_path, file)) for file in files}
    return data_frames


# Function to get all files present in folders and sub folders
@st.cache_data
def get_all_files(directory):
    """
    Recursively get all files in the directory and subdirectories with the given extensions.
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.relpath(os.path.join(root, file), directory))
    return file_list


@st.cache_data
def make_unique_column_names(columns: List[str]) -> List[str]:
    seen = {}
    for i, col in enumerate(columns):
        if col not in seen:
            seen[col] = 1
            columns[i] = col
        else:
            seen[col] += 1
            columns[i] = f"{col}_{seen[col]}"
    return columns


# Function to load retrieval experiment data with caching to optimize performance
@st.cache_data
def load_run_data(file_path: Union[str, Path]) -> pd.DataFrame:
    # Ensure file_path is a string and get the file name without extension
    file_path = str(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Define expected columns
    expected_columns = ["query_id", "iteration", "doc_id", "rank", "score", "tag"]

    # Detect delimiter and check for header
    with open(file_path, 'r') as file:
        sample = file.read(1024)
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False

    # Read the first few lines to check the number of columns
    df_sample = pd.read_csv(file_path, nrows=5, delimiter=delimiter, header=None)

    if df_sample.shape[1] == len(expected_columns) and not has_header:
        # Case: No header, correct number of columns
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            header=None,
            names=expected_columns,
            engine="python",
        )
    else:
        # Case: Has header or incorrect number of columns
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            header=0 if has_header else None,
            engine="python",
        )

        # If no header, use the first row as header
        if not has_header:
            df.columns = df.iloc[0]
            df = df.drop(df.index[0]).reset_index(drop=True)

        # Lowercase column names for consistency
        df.columns = df.columns.str.lower()

        # Map variations of column names
        column_mapping = {
            'qid': 'query_id',
            'q_id': 'query_id',
            'queryid': 'query_id',
            'queryID': 'query_id',
            'docno': 'doc_id',
            'doc_no': 'doc_id',
            'document_no': 'doc_id',
            'doc_rank': 'rank',
            'ranking': 'rank',
            'rank_pos': 'rank',
            'relevance_score': 'score',
            'rel_score': 'score',
            'document_identifier': 'doc_id',
            'docid': 'doc_id_to_drop',  # We'll drop this column later
            'doc_id': 'doc_id_to_drop',  # We'll drop this column later
            'document_id': 'doc_id_to_drop',  # We'll drop this column later
            'label': 'tag'
        }

        # Rename columns based on mapping
        df = df.rename(columns=column_mapping)

        # Drop unnecessary columns
        columns_to_drop = ['query', 'doc_id_to_drop']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Ensure all required columns are present
        for col in expected_columns:
            if col not in df.columns:
                df[col] = "Q0"

        # Ensure unique column names
        df.columns = make_unique_column_names(list(df.columns))

        # Ensure columns are in the correct order
        df = df.reindex(columns=expected_columns, fill_value="MISSING")

    # Convert data types
    df['query_id'] = df['query_id'].astype(str)
    df['iteration'] = df['iteration'].astype(str)
    df['doc_id'] = df['doc_id'].astype(str)
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # If 'tag' column is empty or all values are 'MISSING', fill it with the file name
    if df['tag'].isna().all() or (df['tag'] == '').all() or (df['tag'] == 'Q0').all():
        df['tag'] = file_name

    return df


# Function to load qrels data with caching
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
# Function to load queries. Supported formats: csv, txt, xml
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

    if "queries" in st.session_state:
        st.write(f"Current queries: {st.session_state['queries']}")
    else:
        st.write("No queries available")
