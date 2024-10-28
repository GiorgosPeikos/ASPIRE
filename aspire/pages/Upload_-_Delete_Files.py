import os
import streamlit as st
from utils.data_handler import get_all_files, refresh_file_list

st.markdown(
    """<div style="text-align: center;"><h1>Data Management<h1></div>""",
    unsafe_allow_html=True,
)

# Uploading the retrieval experiments for evaluation
st.header("Upload Retrieval Experiments")
runs = st.file_uploader(
    "Upload Retrieval Experiments (TREC Format). The file's name will be its identifier.",
    type=["txt", "csv"],
    accept_multiple_files=True,
)
st.markdown(
    """
    <i>Expected format: query_id, iteration (i.e. Q0), doc_id, rank, score, experiment_id</i>
    """,
    unsafe_allow_html=True,
)

st.divider()
# Uploading the Qrels files
st.header("Upload Qrels")
qrels = st.file_uploader(
    "Upload Qrels File. The file's name will be its identifier. ", type=["txt", "csv"]
)
st.markdown(
    """
    <i>Expected format: query_id, iteration (i.e. Q0), doc_id, relevance (without header row)</i>
    """,
    unsafe_allow_html=True,
)

st.divider()
# Uploading the queries associated with the previous files
st.header("Upload Queries")
queries = st.file_uploader(
    "Upload Queries File. The file's name will be its identifier. ",
    type=["txt", "csv", "xml", "xlsx", "tsv"],
)
st.markdown(
    """
    <i>Expected format: query_id, query</i>
    """,
    unsafe_allow_html=True,
)


if st.button("Upload Files"):
    saved_files = []

    st.write("Current Working Directory:", os.getcwd())

    if runs is not None:
        runs_folder = "retrieval_experiments/retrieval_runs"
        os.makedirs(runs_folder, exist_ok=True)

        for run in runs:
            run_file_path = os.path.join(runs_folder, run.name)

            with open(run_file_path, "wb") as f:
                f.write(run.getvalue())
            saved_files.append(run.name)

            st.session_state["runs"] = run.name

    if qrels is not None:
        # Define the folder to save the QREL file
        qrels_folder = "retrieval_experiments/qrels/"
        os.makedirs(qrels_folder, exist_ok=True)

        # Define file path using the original file name
        qrels_file_path = os.path.join(qrels_folder, qrels.name)

        # Save the QREL file
        with open(qrels_file_path, "wb") as f:
            f.write(qrels.getvalue())
        saved_files.append(qrels.name)
        st.session_state["qrels"] = qrels.name

    if queries is not None:
        # Define the folder to save the queries file
        queries_folder = "retrieval_experiments/queries/"
        os.makedirs(queries_folder, exist_ok=True)

        # Define file path using the original file name
        queries_file_path = os.path.join(queries_folder, queries.name)

        # Save the Queries file
        with open(queries_file_path, "wb") as f:
            f.write(queries.getvalue())
        saved_files.append(queries.name)
        st.session_state["qrels"] = queries.name

    if saved_files:
        st.success(f'Files saved: {", ".join(saved_files)}')
    else:
        st.error("No files uploaded. Please upload at least one file.")

st.divider()
st.markdown("<h2 style=color:red;>Delete Files</h2>", unsafe_allow_html=True)

# Refresh the file list
all_files = refresh_file_list()

# Create a dictionary to map file names to their relative paths
file_dict = {os.path.basename(file): file for file in all_files}

if all_files:
    selected_files = st.multiselect("Select files to delete", list(file_dict.keys()))

    # Delete Selected Files
    if st.button("Delete selected files"):
        for file_name in selected_files:
            relative_path = file_dict[file_name]
            file_path = os.path.join("retrieval_experiments/", relative_path)
            os.remove(file_path)
            st.write(f"Deleted {relative_path}")

        # Refresh the file list after deletion
        all_files = refresh_file_list()
        file_dict = {os.path.basename(file): file for file in all_files}
        st.success("File list updated after deletion.")
        st.rerun()
else:
    st.write("No files to display.")
