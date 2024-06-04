import streamlit as st
import pandas as pd

from utils.ui import single_run_selector, query_selector
from utils.data import print_session_data
import os
from utils.evaluation_measures import (
    return_available_measures,
    evaluate_single_run,
    evaluate_single_run_custom,
    per_query_evaluation,
    good_bad_queries,
)
from utils.data import load_run_data, load_qrel_data
from utils.plots import create_evaluation_plot, plot_queries

print_session_data()


def initialize_results():
    # Standard, extra, custom, and query results dictionaries are initialized
    if "results_standard" not in st.session_state:
        st.session_state["results_standard"] = {}
    if "results_extra" not in st.session_state:
        st.session_state["results_extra"] = {}
    if "results_custom" not in st.session_state:
        st.session_state["results_custom"] = {}
    if "results_query" not in st.session_state:
        st.session_state["results_query"] = {}
    if "saved_queries" not in st.session_state:
        st.session_state["saved_queries"] = {}
    if "selected_queries" not in st.session_state:
        st.session_state["selected_queries"] = {}


query_selector()
single_run_selector()


if "selected_run" in st.session_state:
    run_path = st.session_state["selected_run"]
    qrel_path = st.session_state["selected_qrels"]
    run = load_run_data(run_path)
    qrels = load_qrel_data(qrel_path)
    relevance_max = qrels["relevance"].max()  # Get maximum relevance level

    # Displaying the experiment's name
    filename_with_suffix = os.path.basename(run_path)
    filename_without_suffix = os.path.splitext(filename_with_suffix)[0]
    st.markdown(
        f"""<div style="text-align: center;"><h1><u>  Evaluating the <span style="color:red;"> <u>{filename_without_suffix} </u></span> Experiment </u> </h1></div>""",
        unsafe_allow_html=True,
    )

    st.header("Mean Performance Evaluation")

    st.sidebar.subheader("Additional settings")
    # Slider for relevance threshold
    relevance_threshold = st.sidebar.slider(
        "Select from the Available Relevance Thresholds (Slide)",
        min_value=1,
        max_value=int(relevance_max),
        value=1,
    )
    if "prev_relevance_threshold" not in st.session_state:
        st.session_state.prev_relevance_threshold = relevance_threshold
    if relevance_threshold != st.session_state.prev_relevance_threshold:
        st.session_state["results_standard"] = {}
        st.session_state["results_extra"] = {}
        st.session_state.prev_relevance_threshold = relevance_threshold

    freq_measures, rest_measures, custom_user = return_available_measures()
    initialize_results()
    freq_measures_results = {}
    for measure_name in freq_measures:
        freq_measures_results[measure_name] = evaluate_single_run(
            qrels, run, measure_name, relevance_threshold
        )
    st.dataframe(pd.DataFrame([freq_measures_results]))

    num_columns = 6
    # Handling additional measures
    if st.button("More Measures", key="more_measures"):
        # Toggles a counter in the session state each time the 'More Measures' button is clicked.
        # This counter is used to track how many times the button has been pressed.
        st.session_state["more_measures_clicks"] = (
            st.session_state.get("more_measures_clicks", 0) + 1
        )

    # Checks if the 'More Measures' button has been clicked an odd number of times
    if st.session_state.get("more_measures_clicks", 0) % 2 != 0:
        # Iterates over the list of additional measures to create buttons for each
        for i in range(0, len(rest_measures), num_columns):
            columns = st.columns(
                num_columns
            )  # Creates a grid layout with a specified number of columns
            for j in range(num_columns):
                index = i + j
                if index < len(
                    rest_measures
                ):  # Ensures the index is within the bounds of the list
                    with columns[j]:
                        measure_name = rest_measures[index]
                        if st.button(measure_name):
                            # Toggles the additional measure on or off in the session state
                            if measure_name in st.session_state["results_extra"]:
                                del st.session_state["results_extra"][measure_name]
                            else:
                                measure_result = evaluate_single_run(
                                    qrels, run, measure_name, relevance_threshold
                                )
                                st.session_state["results_extra"][
                                    measure_name
                                ] = measure_result
                        if measure_name in st.session_state["results_extra"]:
                            # Displays the result of the selected additional measure
                            st.markdown(
                                f"<h2 style='text-align: center;'>{st.session_state['results_extra'][measure_name]}</h2>",
                                unsafe_allow_html=True,
                            )

    # Section for custom measure input
    st.markdown("<h6> Custom Measure Estimation </h6>", unsafe_allow_html=True)
    # User input for custom cutoffs within a form
    with st.form(key="custom_cutoff_form"):
        selected_measure = st.selectbox(
            "Select a measure", custom_user
        )  # Dropdown to select a measure
        cutoff = st.number_input(
            "Enter cutoff value", min_value=1, value=30, max_value=1000, step=1
        )  # Input for cutoff value
        relevance_level = st.number_input(
            "Enter relevance level", min_value=1, max_value=relevance_max, step=1
        )  # Input for relevance level
        submit_button_custom = st.form_submit_button(
            label="Evaluate with Custom Cutoff"
        )  # Button to submit the form

    # Handling the submission of the custom measure form
    if submit_button_custom:
        # Executes the selected measure with user-defined cutoffs
        try:
            parsed_metric, custom_cutoff_result = evaluate_single_run_custom(
                qrels, run, selected_measure, cutoff, relevance_level
            )
            st.session_state["results_custom"][
                f"{selected_measure}(rel={relevance_level})@{cutoff}"
            ] = custom_cutoff_result
            # Displays the result for the custom cutoff measure if available
            custom_measure_key = f"{selected_measure}(rel={relevance_level})@{cutoff}"
            if custom_measure_key in st.session_state["results_custom"]:
                st.markdown(
                    f"<h2 style='text-align: center;'>{custom_measure_key} = {st.session_state['results_custom'][custom_measure_key]}</h2>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            # Displays an error message in case of an exception during evaluation
            st.error(f"Error evaluating custom cutoff: {e}")

    # Header for the 'Per Query Analysis' section
    st.header("Per Query Analysis")

    # Initialize an empty list in the session state to store per-query analysis results, if it doesn't already exist
    if "per_query_analysis_results" not in st.session_state:
        st.session_state["per_query_analysis_results"] = []

    # Creating a form for user inputs specific to per query analysis
    with st.form(key="per_query_analysis"):
        # Dropdown for users to select a measure from a predefined list
        selected_measure_query = st.selectbox("Select a measure", custom_user)

        # Input for users to enter a cutoff value, with specified minimum, default, and maximum values
        cutoff_query = st.number_input(
            "Enter cutoff value", min_value=1, value=25, max_value=1000, step=1
        )

        # Input for users to specify a relevance level, within a defined range
        relevance_level_query = st.number_input(
            "Enter relevance level", min_value=1, max_value=relevance_max, step=1
        )

        # Button to submit the form
        submit_button_query = st.form_submit_button(label="Per Query Score Analysis")

        show_queryperf = st.form_submit_button(label="Show Good/Bad Performing Queries")

    # Handling the form submission
    if submit_button_query:
        try:
            # Evaluating the selected measure with user-defined cutoffs and relevance level for per query analysis
            parsed_metric_query, custom_cutoff_result_query = per_query_evaluation(
                qrels, run, selected_measure_query, cutoff_query, relevance_level_query
            )

            # Creating a unique identifier for the graph based on the measure and relevance level
            graph_identifier = (selected_measure_query, relevance_level_query)

            # Searching for an existing graph in the session state that matches the current measure and relevance level
            existing_index = None
            for i, (metric_info, _) in enumerate(
                st.session_state["per_query_analysis_results"]
            ):
                if (
                    metric_info["measure"],
                    metric_info["relevance_level"],
                ) == graph_identifier:
                    existing_index = i
                    break

            # Creating a dictionary with the new graph information
            new_graph_info = {
                "measure": selected_measure_query,
                "relevance_level": relevance_level_query,
                "cutoff": cutoff_query,
            }

            # Updating or appending the graph in the session state
            if existing_index is not None:
                # Replace existing graph if the cutoff is different
                if (
                    st.session_state["per_query_analysis_results"][existing_index][0][
                        "cutoff"
                    ]
                    != cutoff_query
                ):
                    st.session_state["per_query_analysis_results"][existing_index] = (
                        new_graph_info,
                        custom_cutoff_result_query,
                    )
            else:
                # Append new graph if it doesn't already exist
                st.session_state["per_query_analysis_results"].append(
                    (new_graph_info, custom_cutoff_result_query)
                )

        except Exception as e:
            # Displaying an error message in case of an exception during evaluation
            st.error(f"Error evaluating custom cutoff: {e}")

    # Iterating through the per-query analysis results and displaying them
    for i in range(0, len(st.session_state["per_query_analysis_results"]), 2):
        cols = st.columns(2)

        for j in range(2):  # Handling two graphs per iteration
            index = i + j
            if index < len(st.session_state["per_query_analysis_results"]):
                metric_info, result = st.session_state["per_query_analysis_results"][
                    index
                ]
                # Creating a title for each graph
                graph_title = f"{metric_info['measure']}(rel={metric_info['relevance_level']})@{metric_info['cutoff']}"

                # In each column, creating a row for the graph title and a button to remove the graph
                with cols[j]:
                    title_col, button_col = st.columns([0.10, 0.02], gap="small")
                    with title_col:
                        st.markdown(
                            f"Evaluation Measure: {graph_title}", unsafe_allow_html=True
                        )
                    with button_col:
                        # Button to remove the graph from the display and session state
                        if st.button("❌", key=f"close_{index}", help="Remove graph"):
                            del st.session_state["per_query_analysis_results"][index]
                            st.rerun()

                    # Rendering the graph plot below the title and close button
                    create_evaluation_plot(result, graph_title)
                    if show_queryperf:
                        scores, ids, median, average = good_bad_queries(result)
                        plot_queries(scores, ids, median, average, graph_title)


else:
    st.error("Errors in Calculations. No run selected.")


st.sidebar.markdown(
    "Further [Details](https://ir-measur.es/en/latest/measures.html) on evaluation measures"
)
