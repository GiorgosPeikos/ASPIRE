import streamlit as st

from utils.ui import load_css


def main():
    st.set_page_config(layout="wide")
    load_css("css/styles.css")

    st.title("Interactive Dashboard for IR Evaluation")
    st.write(
        "This application allows you to analyze retrieval experiments, "
        "compare multiple retrieval experiments, and create various visualizations."
    )


if __name__ == "__main__":
    main()
