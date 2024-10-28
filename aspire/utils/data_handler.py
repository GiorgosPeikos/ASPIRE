import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Union
import pandas as pd
import streamlit as st


# Function to get all files present in folders and sub folders
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
    with open(file_path, "r") as file:
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
            "qid": "query_id",
            "q_id": "query_id",
            "queryid": "query_id",
            "queryID": "query_id",
            "docno": "doc_id",
            "doc_no": "doc_id",
            "document_no": "doc_id",
            "doc_rank": "rank",
            "ranking": "rank",
            "rank_pos": "rank",
            "relevance_score": "score",
            "rel_score": "score",
            "document_identifier": "doc_id",
            "docid": "doc_id_to_drop",  # We'll drop this column later
            "doc_id": "doc_id_to_drop",  # We'll drop this column later
            "document_id": "doc_id_to_drop",  # We'll drop this column later
            "label": "tag",
        }

        # Rename columns based on mapping
        df = df.rename(columns=column_mapping)

        # Drop unnecessary columns
        columns_to_drop = ["query", "doc_id_to_drop"]
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
    df["query_id"] = df["query_id"].astype(str)
    df["iteration"] = df["iteration"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # If 'tag' column is empty or all values are 'MISSING', fill it with the file name
    if df["tag"].isna().all() or (df["tag"] == "").all() or (df["tag"] == "Q0").all():
        df["tag"] = file_name

    return df


# Function to load qrels data with caching
@st.cache_data
def load_qrel_data(qrel_path):
    file_extension = os.path.splitext(qrel_path)[1].lower()

    if file_extension in [".txt", ".csv", ".tsv"]:
        try:
            # Read the entire file content
            with open(qrel_path, 'r') as file:
                content = file.read()

            # Split the content into lines
            lines = content.strip().split('\n')

            # Determine the number of fields
            num_fields = len(lines[0].split())

            if num_fields == 3:
                # Case: query_id, doc_id, relevance
                df = pd.DataFrame([line.split(None, 2) for line in lines],
                                  columns=['query_id', 'doc_id', 'relevance'])
                df['iteration'] = 'Q0'
            elif num_fields == 4:
                # Case: query_id, iteration, doc_id, relevance
                df = pd.DataFrame([line.split(None, 3) for line in lines],
                                  columns=['query_id', 'iteration', 'doc_id', 'relevance'])
            else:
                # Case: Flexible parsing for query_id and doc_id with spaces
                data = []
                for line in lines:
                    parts = line.split()
                    if len(parts) < 3:
                        continue  # Skip invalid lines
                    relevance = parts[-1]
                    doc_id = parts[-2]
                    query_id = ' '.join(parts[:-2])
                    data.append([query_id, doc_id, relevance])
                df = pd.DataFrame(data, columns=['query_id', 'doc_id', 'relevance'])
                df['iteration'] = 'Q0'

            # Ensure columns are in the correct order
            df = df[['query_id', 'iteration', 'doc_id', 'relevance']]

            # Convert relevance to numeric, replacing non-numeric values with 0
            df['relevance'] = pd.to_numeric(df['relevance'], errors='coerce').fillna(0).astype(int)

            # Remove any leading/trailing whitespace from string columns
            for col in ['query_id', 'iteration', 'doc_id']:
                df[col] = df[col].str.strip()

            return df

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.error("Please check your QREL file for inconsistencies.")
            return pd.DataFrame()  # Return an empty DataFrame if there's an error

    else:
        raise ValueError("Unsupported file format for QREL data")


@st.cache_data
def load_query_data(query_path):
    file_extension = os.path.splitext(query_path)[1].lower()

    if file_extension in [".csv", ".tsv", ".txt"]:
        return read_delimited_file(query_path)
    elif file_extension == ".xlsx":
        return read_excel_file(query_path)
    elif file_extension == ".xml":
        return read_xml_file(query_path)
    else:
        raise ValueError("Unsupported file format")


def read_delimited_file(file_path):
    try:
        with open(file_path, 'r') as file:
            sample = file.read(1024)
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(sample)
            dialect = sniffer.sniff(sample)

        if file_path.endswith('.txt'):
            # For .txt files, try common delimiters
            delimiters = ['\t', ',']
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(
                        file_path,
                        sep=delimiter,
                        header=0 if has_header else None,
                        names=['query_id', 'query_text'] if not has_header else None,
                        dtype={"query_id": str, "query_text": str},
                        engine='python'
                    )
                    if len(df.columns) == 2:
                        return df
                except:
                    continue
            raise ValueError("Could not determine delimiter for .txt file")
        else:
            return pd.read_csv(
                file_path,
                dialect=dialect,
                header=0 if has_header else None,
                names=['query_id', 'query_text'] if not has_header else None,
                dtype={"query_id": str, "query_text": str},
                engine='python'
            )
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.error("Please check your file for inconsistencies.")
        return pd.DataFrame()


def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path, dtype={"query_id": str, "query_text": str})
        if len(df.columns) != 2:
            raise ValueError("Excel file must have exactly two columns: query_id and query_text")
        df.columns = ['query_id', 'query_text']
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return pd.DataFrame()


def read_xml_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        query_ids = []
        query_texts = []

        for topic in root.findall(".//topic"):
            query_id = topic.get("number")
            query_text = "".join(topic.itertext()).strip()
            query_ids.append(query_id)
            query_texts.append(query_text)

        return pd.DataFrame({"query_id": query_ids, "query_text": query_texts})
    except Exception as e:
        st.error(f"Error reading XML file: {e}")
        return pd.DataFrame()


def refresh_file_list():
    return get_all_files("retrieval_experiments/")
