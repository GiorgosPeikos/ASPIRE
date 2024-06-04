import streamlit as st
import pandas as pd
import os


def read_data(folder_path: str):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    data_frames = {file: pd.read_csv(os.path.join(folder_path, file)) for file in files}
    return data_frames


def load_css(file_name: str):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def print_session_data() -> None:
    if "qrels" in st.session_state:
        st.write(f"Loaded qrels: {st.session_state['qrels']}")
    else:
        st.write("No qrels available")

    if "runs" in st.session_state:
        st.write(f"Current runs: {st.session_state['runs']}")
    else:
        st.write("No runs available")
