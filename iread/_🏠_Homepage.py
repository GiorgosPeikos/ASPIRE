import streamlit as st
from utils.ui import load_css


def main():
    st.set_page_config(layout="wide", page_title="IR Evaluation Dashboard")
    load_css("css/styles.css")

    st.markdown("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(to bottom, #f0f2f6, #e1e5eb); border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h1 style="color: #2c3e50; margin-bottom: 5px; font-size: 2.5em;">
                ASPIRE
            </h1>
            <h3 style="color: #34495e; margin-top: 0; margin-bottom: 20px; font-weight: normal;">
                Assistive System for Performance Evaluation in IR
            </h3>
            <hr style="width: 50%; margin: 20px auto; border: none; border-top: 2px solid #3498db;">
            <h4 style="color: #1f77b4; font-style: italic; margin-bottom: 0;">
                🚀 Empowering researchers to see beyond the numbers! 🚀
            </h4>
        </div>
        """, unsafe_allow_html=True)

    # Overview section
    st.header("Overview")
    st.write(
        "The Assistive System for Performance Evaluation in Information Retrieval (ASPIRE) is a tool designed for "
        "researchers and practitioners in the field of Information Retrieval. It provides a "
        "user-friendly interface to analyze and compare the performance of different retrieval "
        "systems or approaches."
    )

    # Available pages section
    st.header("Available Functionalities")
    col1, col2 = st.columns(2)

    pages = [
        ("📊 Experiment Performance Report", "Evaluate experiments using standard IR metrics. Compare multiple runs, visualize performance, and conduct statistical significance testing."),
        ("🔍 Query-based Report", "Analyze performance on a per-query basis. Identify consistently performing queries, compare experiments, and visualize query-level performance."),
        ("📝 Query Text-based Report", "Examine the relationship between query characteristics and performance. Analyze query length impact, create word clouds, and visualize query similarity."),
        ("📚 Query Collection-based Report", "Analyze relevance judgments distribution, identify documents with multiple query relevance, and visualize retrieved document rankings."),
        ("🔮 Query Performance Prediction vs Query Performance Report", "⚠️ Coming Soon! 🚧"),
        ("🌐 Multidimensional Relevance Experiment Performance Report", "⚠️ Coming Soon! 🚧"),
        ("📁 Upload - Delete Files", "Manage your experimental data files. Upload retrieval runs, qrels, and query files, or delete existing files."),
    ]

    for i, (page, description) in enumerate(pages):
        with col1 if i % 2 == 0 else col2:
            with st.expander(page):
                st.write(description)
                if page == "Upload - Delete Files":
                    st.write("- Upload and manage TREC format run files, qrels, and query files")
                    st.write("- Supports various file formats including txt, csv, and xml")
                elif page == "Experiment Performance Report":
                    st.write("- Calculate and compare standard IR metrics across multiple runs")
                    st.write("- Visualize overall retrieval characteristics")
                    st.write("- Perform statistical significance testing with multiple correction methods")
                    st.write("- Generate precision-recall curves")
                elif page == "Query-based Report":
                    st.write("- Analyze per-query performance across experiments")
                    st.write("- Identify queries with consistent performance or large gaps")
                    st.write("- Compare experiments against baselines or thresholds")
                elif page == "Query Text-based Report":
                    st.write("- Analyze query performance based on query length")
                    st.write("- Generate word clouds based on query relevance")
                    st.write("- Visualize query similarity in 2D and 3D spaces")
                elif page == "Query Collection-based Report":
                    st.write("- Analyze relevance judgment distribution across queries")
                    st.write("- Identify documents with relevance judgments for multiple queries")
                    st.write("- Visualize document rankings and relevance across experiments")

    # Getting Started section
    st.header("Getting Started")
    st.markdown("""
    1. Begin by uploading your experimental data using the "Upload - Delete Files" page.
    2. Navigate to the desired analysis tool from the sidebar.
    3. Follow the on-screen instructions to configure your analysis.
    4. Interact with the generated visualizations and tables to gain insights into your IR experiments.
    5. Download the created graphs.
    6. Download the created analysis as PDF.
    """)

    st.markdown("---")
    # Vision section
    st.header("🎯 Our Vision")

    st.markdown("""
    ASPIRE is more than just a tool!
    Our vision is to:

    - Create a comprehensive repository of inspiring and best practices in IR evaluation, collected over years of research and experimentation.

    - Enable researchers to evaluate and analyze retrieval experiments from various publications, going beyond the tables reported in papers.

    - Take a significant step towards greater reproducibility and transparency in the field of Information Retrieval.

    By providing a platform for in-depth analysis of published results, we aim to foster a more open, 
    collaborative, and rigorous research environment. ASPIRE represents a low-level yet crucial step 
    towards reproducibility and transparency in IR experimentation.
    """)

    # Footer
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <p><strong>For more information, contributions, or support, please contact:</strong></p>
            <p>Georgios Peikos: <span style="color: #1f77b4;">georgios.peikos@unimib.it</span></p>
            <p>Wojciech Kusa: <span style="color: #1f77b4;">wojciech.kusa@tuwien.ac.at</span></p>
            <a href="https://github.com/GiorgosPeikos/IREAD" target="_blank" style="text-decoration: none;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" height="30" style="vertical-align: middle;">
                <span style="vertical-align: middle; margin-left: 10px; margin-top: 40px;">Visit our GitHub repository</span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()