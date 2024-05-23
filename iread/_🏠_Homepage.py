import streamlit as st
from utils.utils import load_css

def main():
    st.set_page_config(layout="wide")
    load_css('iread/css/styles.css')


    with st.sidebar:
        if st.button("Main Page"):
            st.session_state['page'] = 'Run Selection and Information'
        if st.button("File Submissions"):
            st.session_state['page'] = 'Submit Experiments for Analysis'


if __name__ == "__main__":
    main()
