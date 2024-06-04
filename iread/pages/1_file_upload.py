import os

import streamlit as st

from utils.data import print_session_data

print_session_data()

st.header("Upload the retrieval experiments in TREC Format")
runs = st.file_uploader(
    "Upload Retrieval Experiments (TREC Format). The file's name will be its identifier.",
    type=["txt", "csv"],
    accept_multiple_files=True,
)
st.markdown(
    """
    <i>Expected columns: query_id, iteration (i.e. Q0), doc_id, rank, score, experiment_id</i>
    """,
    unsafe_allow_html=True,
)

st.header("Upload the Qrels file in TREC Format")
qrels = st.file_uploader(
    "Upload Qrels File. The file's name will be its identifier. ", type=["txt", "csv"]
)
st.markdown(
    """
    <i>Expected columns: query_id, iteration (i.e. Q0), doc_id, relevance (without header row)</i>
    """,
    unsafe_allow_html=True,
)


if st.button("Upload Files"):
    saved_files = []

    st.write("Current Working Directory:", os.getcwd())

    if runs is not None:
        runs_folder = "../retrieval_experiments/retrieval_runs"
        os.makedirs(runs_folder, exist_ok=True)

        for run in runs:
            run_file_path = os.path.join(runs_folder, run.name)

            with open(run_file_path, "wb") as f:
                f.write(run.getvalue())
            saved_files.append(run.name)

            st.session_state["runs"] = run.name

    if qrels is not None:
        # Define the folder to save the QREL file
        qrels_folder = "../retrieval_experiments/qrels/"
        os.makedirs(qrels_folder, exist_ok=True)

        # Define file path using the original file name
        qrels_file_path = os.path.join(qrels_folder, qrels.name)

        # Save the QREL file
        with open(qrels_file_path, "wb") as f:
            f.write(qrels.getvalue())
        saved_files.append(qrels.name)
        st.session_state["qrels"] = qrels.name

    if saved_files:
        st.success(f"Files saved: {', '.join(saved_files)}")
    else:
        st.error("No files uploaded. Please upload at least one file.")
