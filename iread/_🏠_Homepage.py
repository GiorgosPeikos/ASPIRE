import streamlit as st


def main():
    st.set_page_config(layout="wide")

    with st.sidebar:
        if st.button("Main Page"):
            st.session_state['page'] = 'Run Selection and Information'
        if st.button("File Submissions"):
            st.session_state['page'] = 'Submit Experiments for Analysis'


if __name__ == "__main__":
    main()
