import time

import streamlit as st

# Page config
st.set_page_config(page_title="Coming Soon!", page_icon="🚧", layout="centered")

# Custom CSS
st.markdown(
    """
<style>
    .stAlert {
        background-color: #f0f2f6;
        border: 2px solid #ffd700;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown("<p class='big-font'>🚧 UNDER Development 🚧</p>", unsafe_allow_html=True)

# Progress bar
progress_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.05)
    progress_bar.progress(percent_complete + 1)
