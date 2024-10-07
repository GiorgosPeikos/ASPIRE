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

    # Read the file content
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Check if the file has a header
    first_line = lines[0].strip().split()
    has_header = len(first_line) == len(expected_columns) and not first_line[0].startswith('query_')

    # Parse the lines
    data = []
    for line in lines[1:] if has_header else lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            query_id, iteration, doc_id, rank, score, *tag = parts
            tag = ' '.join(tag)  # Join the remaining parts as the tag
            data.append([query_id, iteration, doc_id, rank, score, tag])

    # Create DataFrame
    df = pd.DataFrame(data, columns=expected_columns)

    # Convert data types
    df["query_id"] = df["query_id"].astype(str)
    df["iteration"] = df["iteration"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # If 'tag' column is empty, fill it with the file name
    if df["tag"].isna().all() or (df["tag"] == "").all():
        df["tag"] = file_name

    return df


# Function to load qrels data with caching
@st.cache_data
def load_qrel_data(qrel_path):
    # Define expected columns
    expected_columns = ["query_id", "doc_id", "relevance"]

    # Determine the file extension
    file_extension = os.path.splitext(qrel_path)[1].lower()

    if file_extension in [".txt", ".csv", ".tsv"]:
        try:
            # Read a sample of the file
            with open(qrel_path, 'r') as csvfile:
                sample = csvfile.read(1024)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)

            # Use the sniffed information to read the CSV
            df = pd.read_csv(
                qrel_path,
                dialect=dialect,
                header=None,
                names=expected_columns,
                dtype={"query_id": str, "doc_id": str, "relevance": str},
                engine="python",
            )

            # Convert relevance to numeric, replacing non-numeric values with 0
            df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce").fillna(0).astype(int)
            return df

        except (pd.errors.ParserError, csv.Error) as e:
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
            query_id = str(topic.get("number"))
            query_text = "".join(topic.itertext()).strip()
            query_ids.append(query_id)
            query_texts.append(query_text)

        return pd.DataFrame({"query_id": query_ids, "query_text": query_texts})
    except Exception as e:
        st.error(f"Error reading XML file: {e}")
        return pd.DataFrame()


def refresh_file_list():
    return get_all_files("../retrieval_experiments/")
