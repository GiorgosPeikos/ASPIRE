# ASPIRE: Assistive System for Performance Evaluation in IR

🚀 Empowering researchers to see beyond just numbers! 🚀


## Overview
This Interactive Dashboard for IR Evaluation is a tool designed for researchers and practitioners in the field of Information Retrieval (IR). It provides a user-friendly interface to analyze and compare the performance of different retrieval systems or approaches. Built with Streamlit, this dashboard offers a range of analyses and visualizations to help users gain insights into their IR experiments.

## Features

### Data Management
- Upload retrieval experiment files (TREC format)
- Upload qrels (relevance judgments) files
- Upload query files (supports txt, csv, and xml formats)
- Delete uploaded files through the interface

### Analysis Tools
1. **Overall Retrieval Characteristics**
   - Evaluate experiments using standard IR metrics (e.g., MAP, nDCG, Precision, Recall)
   - Visualize performance across multiple runs
   - Perform statistical significance testing between experiments
   - Apply multiple testing corrections (e.g., Bonferroni, Holm, Holm-Sidak)
   - Generate precision-recall curves

2. **Query-based Analysis**
   - Analyze performance on a per-query basis
   - Compare query performance across different experiments
   - Identify consistently performing queries and those with large performance gaps

3. **Query Text-based Analysis**
   - Examine the relationship between query length and performance
   - Visualize query performance using scatter plots and moving averages
   - Perform bucketed analysis (equal-width and equal-frequency)
   - Create word clouds based on query relevance
   - Visualize query similarity in 2D and 3D spaces using dimensionality reduction techniques

4. **Collection-based Analysis**
   - Analyze relevance judgments distribution
   - Identify easy and hard queries based on relevance assessments


## Installation

1. Clone the repository: https://github.com/GiorgosPeikos/IREAD.git 

2. Install requirements: 

	```
	pip install -r requirements.txt
	```
3. Start the Streamlit Application:

	```
	streamlit run iread/_🏠_Homepage.py
	```
	
4. The dashboard will open in your default web browser. Use the sidebar to navigate between different tools:
- Upload - Delete Files
- Experiment Performance Report
- Query-based Report
- Query Text-based Report
- Query Collection-based Report

5. Begin by uploading your experimental data using the "Upload - Delete Files" page.

6. Navigate to the desired analysis tool and follow the on-screen instructions to configure your analysis.

7. Interact with the generated visualizations and tables to gain insights into your IR experiments.

## File Structure

- `_🏠_Homepage.py`: Main entry point for the Streamlit app
- `Upload_-_Delete_Files.py`: Interface for managing experimental data files
- `Experiment_Performance_Report.py`: Overall performance analysis script
- `Experiment_Performance_Query-based_Report.py`: Query-based analysis script
- `Experiment_Performance_Query_Text-based_Report.py`: Text-based query analysis script
- `Experiment_Performance_Query_Collection-based_Report.py`: Collection-based analysis script
- `utils/`: Directory containing utility functions and modules
- `data_handler.py`: Functions for loading and processing data files
- `eval_core.py`: Core evaluation functions and metric definitions
- `eval_multiple_exp.py`: Functions for evaluating multiple experiments
- `eval_per_query.py`: Per-query evaluation functions
- `eval_query_collection.py`: Query collection analysis functions
- `eval_query_text_based.py`: Text-based query analysis functions
- `eval_single_exp.py`: Single experiment evaluation functions
- `plots.py`: Visualization functions
- `ui.py`: User interface utility functions

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- matplotlib
- scipy
- ir_measures
- scikit-learn
- transformers
- wordcloud
- nltk

For a complete list of dependencies with version numbers, refer to the `requirements.txt` file.

## Contributing

Contributions to improve the dashboard or add new features are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## Contact

[Georgios Peikos] - [georgios.peikos@unimib.it]

[Wojciech Kusa] - [wojciech.kusa@tuwien.ac.at]

Project Link: https://github.com/GiorgosPeikos/IREAD.git 

## Acknowledgments

- [ir_measures](https://github.com/terrierteam/ir_measures) for providing IR evaluation metrics.
- [Streamlit](https://streamlit.io/) for the web app framework.
- [Plotly](https://plotly.com/) for data visualizations.
- [Matplotlib](https://matplotlib.org/) for data visualization capabilities.
- Tested on Python 3.10
