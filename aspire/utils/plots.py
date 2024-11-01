import math
import random
from collections import defaultdict
import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from utils.eval_query_collection import get_query_rel_judgements
from utils.eval_query_text_based import (query_clf_relevance_assessments,
                                         query_similarity_performance,
                                         remove_stopwords_from_queries)
from utils.eval_single_exp import get_experiment_name
from wordcloud import WordCloud
import numpy as np


def sort_query_ids(query_ids):
    def extract_number(query_id):
        match = re.search(r'\d+', query_id)
        return int(match.group()) if match else float('inf')

    return sorted(query_ids, key=extract_number)


# Function that displays the distribution of ranking position of all retrieved documents based on their relevance label.
@st.cache_data
def plot_dist_of_retrieved_docs(relevance_ret_pos: dict) -> None:
    # Define constants for bucket ranges
    bucket_ranges = {
        "1": (1, 1),
        "2-10": (2, 10),
        "11-20": (11, 20),
        "21-30": (21, 30),
        "31-40": (31, 40),
        "41-50": (41, 50),
        "51-60": (51, 60),
        "61-70": (61, 70),
        "71-80": (71, 80),
        "81-90": (81, 90),
        "91-100": (91, 100),
        "101-200": (101, 200),
        "200+": (201, 999),  # Handle values greater than 200
    }

    # Initialize buckets dictionary dynamically
    thresholds = set(
        key for values in relevance_ret_pos.values() for key in values.keys()
    )
    buckets = {
        threshold: {bucket: 0 for bucket in bucket_ranges.keys()}
        for threshold in thresholds
    }

    # Iterate through the items in relevance_ret_pos dictionary
    for query_id, values in relevance_ret_pos.items():
        for threshold, value in values.items():
            value = int(value)
            for bucket, (start, end) in bucket_ranges.items():
                if start <= value <= end:
                    buckets[threshold][bucket] += 1
                    break  # Break once the bucket is found

    # Combine all data into a single list for sorting and plotting
    all_data = []
    for metric, bucket_counts in buckets.items():
        for bucket, count in bucket_counts.items():
            all_data.append((metric, bucket, count))

    # Sort by bucket ranges
    all_data.sort(key=lambda x: list(bucket_ranges.keys()).index(x[1]))

    # Extract sorted data for plotting
    sorted_buckets = defaultdict(lambda: defaultdict(int))
    for metric, bucket, count in all_data:
        sorted_buckets[metric][bucket] = count

    # Plot all metrics in a single figure using Plotly
    fig = go.Figure()

    x_labels = list(bucket_ranges.keys())
    x_indices = list(range(len(x_labels)))  # Convert range to list
    num_metrics = len(buckets)
    colors = [
        "skyblue",
        "lightgreen",
        "salmon",
        "gold",
    ]  # Different colors for each metric
    width = 0.2  # Width of each bar

    for index, (metric, bucket_counts) in enumerate(sorted_buckets.items()):
        fig.add_trace(
            go.Bar(
                x=[i + (index - num_metrics / 2) * width for i in x_indices],
                y=[bucket_counts[bucket] for bucket in x_labels],
                width=width,
                name=metric,
                marker_color=colors[index % len(colors)],
                hoverinfo="y",
            )
        )

    fig.update_layout(
        title="Distribution of Document Ranking Positions",
        xaxis_title="Rank Position of the 1st Retrieved Document per Relevance Label",
        yaxis_title="Number of Queries",
        xaxis=dict(tickmode="array", tickvals=x_indices, ticktext=x_labels),
        barmode="group",
        legend=dict(
            orientation="h",  # horizontal legend
            yanchor="bottom",  # anchor legend to the bottom
            y=1.02,  # position the legend just below the plot
            xanchor="right",  # anchor legend to the right
            x=1,  # position legend to the right of the plot
        ),
    )
    # Display the plot in Streamlit
    st.plotly_chart(fig)


@st.cache_data
def plot_precision_recall_curve(prec_recall_graphs, relevance_thres):
    """
    This function takes a dictionary containing precision-recall data and plots a precision-recall curve using Plotly.
    The dictionary keys are expected to be in the format 'IPrec(rel=1)@threshold', where 'threshold' is a floating-point
    number representing recall values (0.1, 0.2, ..., 1.0). The corresponding values are the precision values at those
    recall thresholds.

    Parameters:
    - prec_recall_graphs (dict): A dictionary with precision-recall data, where keys are strings in the format
      'IPrec(rel=1)@threshold' and values are floating-point numbers representing precision values.
    """
    # Extracting thresholds and precisions
    thresholds = sorted([float(key.split("@")[1]) for key in prec_recall_graphs.keys()])
    precisions = [
        prec_recall_graphs[f"IPrec(rel={relevance_thres})@{threshold}"]
        for threshold in thresholds
    ]

    # Add a point for recall=0 to start the plot from the y-axis
    if 0.0 not in thresholds:
        thresholds.insert(0, 0.0)
        precisions.insert(
            0, prec_recall_graphs.get(f"IPrec(rel={relevance_thres})@0", 0)
        )

    # Create the plot using Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=precisions,
            mode="lines+markers",
            name="Precision-Recall Curve",
            hoverinfo="y",
            line=dict(shape="linear"),
        )
    )

    fig.update_layout(
        title=f"Precision-Recall Curve (Relevance_threshold={relevance_thres})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


@st.cache_data
# Generate colors dynamically
def generate_colors(n):
    # Predefined list of LaTeX-friendly colors
    predefined_colors = [
        "#E69F00",  # Orange
        "#56B4E9",  # Sky Blue
        "#009E73",  # Bluish Green
        "#0072B2",  # Blue
        "#D55E00",  # Vermilion
        "#CC79A7",  # Reddish Purple
        "#000000",  # Black
        "#E69F00",  # Orange
        "#56B4E9",  # Sky Blue
        "#009E73",  # Bluish Green
        "#0072B2",  # Blue
        "#CC79A7",  # Reddish Purple
    ]

    # If we need more colors than predefined, cycle through the list
    if n > len(predefined_colors):
        # Calculate how many times to repeat the color list
        repeat_count = (n // len(predefined_colors)) + 1
        predefined_colors = predefined_colors * repeat_count

    # Shuffle the colors to ensure a good mix if we're using repeated colors
    random.shuffle(predefined_colors)

    return predefined_colors[:n]


@st.cache_data
def plot_performance_measures_per_q(data):
    # Debugging: Print out the structure of the data
    # st.write("Data structure:")
    # for run in data:
    #     st.write(f"Run: {run}")
    #     for measure in data[run]:
    #         st.write(f"  Measure: {measure}")
    #         st.write(f"    Keys: {data[run][measure].keys()}")
    #         st.write(f"    Data: {data[run][measure]}")

    # Extract measures and runs
    eval_measures = list(set(measure for run in data for measure in data[run]))
    runs = list(data.keys())

    # Create a subplot for each measure
    fig = make_subplots(rows=len(eval_measures), cols=1, subplot_titles=eval_measures, vertical_spacing=0.3)

    # Generate more colors if needed
    colors = plt.cm.rainbow(np.linspace(0, 1, len(runs)))
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in colors]

    # Generate patterns dynamically
    base_patterns = ["", "/", "x", "-", "|", "+", ".", "\\"]
    patterns = [base_patterns[i % len(base_patterns)] for i in range(len(runs))]

    # Get all unique query IDs across all runs and measures, and sort them
    all_query_ids = set()
    for run in runs:
        for measure in eval_measures:
            if measure in data[run] and 'query_id' in data[run][measure]:
                all_query_ids.update(data[run][measure]['query_id'])
            elif measure in data[run]:
                all_query_ids.update(range(len(data[run][measure][measure])))

    sorted_query_ids = sort_query_ids(list(all_query_ids))

    for i, measure in enumerate(eval_measures, start=1):
        for j, run in enumerate(runs):
            if measure in data[run]:
                if 'query_id' in data[run][measure]:
                    query_ids = data[run][measure]['query_id']
                else:
                    query_ids = list(range(len(data[run][measure][measure])))
                measure_values = data[run][measure][measure]

                # Create a mapping of query_id to its index in sorted_query_ids
                query_id_to_index = {qid: idx for idx, qid in enumerate(sorted_query_ids)}

                # Sort the data based on the order in sorted_query_ids
                sorted_indices = [query_id_to_index[qid] for qid in query_ids if qid in query_id_to_index]
                sorted_query_ids_run = [query_ids[i] for i in range(len(query_ids)) if i in sorted_indices]
                sorted_measure_values = [measure_values[i] for i in range(len(measure_values)) if i in sorted_indices]

                fig.add_trace(
                    go.Bar(
                        x=sorted_query_ids_run,
                        y=sorted_measure_values,
                        name=run,
                        marker_color=colors[j],
                        marker_pattern_shape=patterns[j],
                        opacity=0.8,
                        showlegend=(i == 1),  # Only show legend for the first subplot
                        hovertemplate="Query: %{x}<br>" + f"{measure}: " + "%{y:.3f}<br>Run: " + run + "<extra></extra>",
                    ),
                    row=i,
                    col=1,
                )

        # Update y-axis title and x-axis settings
        fig.update_yaxes(title_text=f"{measure} Value", row=i, col=1)
        fig.update_xaxes(
            title_text="Query ID",
            tickmode="array",
            tickvals=sorted_query_ids,
            ticktext=sorted_query_ids,
            tickangle=45,
            row=i,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=300 * len(eval_measures),
        title_text="Performance Measures Across Queries",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, b=50),
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_performance_difference(data):
    # Extract measures and runs
    eval_measures = list(data[list(data.keys())[0]].keys())
    runs = list(data.keys())

    # Find the baseline run
    baseline_run = next(run for run in runs if "(Baseline)" in run)
    other_runs = [run for run in runs if run != baseline_run]

    # Determine the maximum number of queries
    max_queries = max(
        len(data[run][measure][measure]['values'])
        for run in runs for measure in eval_measures
    )

    # Generate colors and patterns
    colors = generate_colors(len(other_runs))
    base_patterns = ["", "/", "x", "-", "|", "+", "."]
    patterns = [base_patterns[i % len(base_patterns)] for i in range(len(other_runs))]

    # Check if any query ID is longer than 5 characters
    long_labels = any(
        len(str(qid)) > 5
        for run in runs
        for measure in eval_measures
        for qid in data[run][measure][measure]['query_ids']
    )

    # Create a subplot for each measure
    fig = make_subplots(
        rows=len(eval_measures),
        cols=1,
        subplot_titles=eval_measures,
        vertical_spacing=0.15,
    )

    for i, measure in enumerate(eval_measures, start=1):
        for j, run in enumerate(other_runs):
            baseline_values = data[baseline_run][measure][measure]['values']
            run_values = data[run][measure][measure]['values']
            query_ids = data[run][measure][measure]['query_ids']

            # Calculate the difference
            diff_values = [
                run_val - baseline_val
                for run_val, baseline_val in zip(run_values, baseline_values)
            ]

            fig.add_trace(
                go.Bar(
                    x=query_ids,
                    y=diff_values,
                    name=run,
                    marker_color=colors[j],
                    marker_pattern_shape=patterns[j],
                    opacity=0.8,
                    showlegend=(i == 1),  # Only show legend for the first subplot
                    hovertemplate="Query: %{x}<br>Difference: %{y:.3f}<br>Run: "
                    + run
                    + "<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=query_ids[0],
            x1=query_ids[-1],
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
            row=i,
            col=1,
        )

        # Update y-axis title
        fig.update_yaxes(title_text=f"{measure} Difference", row=i, col=1)

        # Update x-axis settings
        if long_labels:
            fig.update_xaxes(
                title_text="Query ID",
                tickmode="array",
                tickvals=query_ids,  # Show all queryids
                ticktext=query_ids,
                tickangle=45,  # Rotate labels 90 degrees
                row=i,
                col=1,
            )
        else:
            fig.update_xaxes(
                title_text="Query ID",
                tickmode="array",
                tickvals=query_ids,
                ticktext=query_ids,
                row=i,
                col=1,
            )

    # Update layout
    fig.update_layout(
        height=550 * len(eval_measures),  # Increase height
        title={
            "text": f"""Performance Difference Compared to the Selected Baseline: <span style="color:red;">{baseline_run.replace('(Baseline)', '')}</span>""",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, b=50),

    )
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_performance_and_median_per_experiment(results_per_run):
    # Extract measures and runs
    all_keys = list(results_per_run[list(results_per_run.keys())[0]].keys())
    eval_measures = [measure for measure in all_keys if not measure.startswith("median_")]
    runs = list(results_per_run.keys())

    # Generate colors
    colors = generate_colors(len(runs))

    # Create a plot for each experiment
    for current_exp in runs:
        # Create a subplot for each measure
        fig = make_subplots(
            rows=len(eval_measures),
            cols=1,
            subplot_titles=eval_measures,
            vertical_spacing=0.15,
        )

        for i, measure in enumerate(eval_measures, start=1):
            for j, run in enumerate(runs):
                if measure in results_per_run[run] and measure in results_per_run[run][measure]:
                    values = results_per_run[run][measure][measure]['values']
                    query_ids = results_per_run[run][measure][measure]['query_ids']

                    fig.add_trace(
                        go.Bar(
                            x=query_ids,
                            y=values,
                            name=run,
                            marker_color=colors[j],
                            showlegend=(i == 1),  # Only show legend for the first subplot
                            hovertemplate="Query: %{x}<br>Value: %{y:.3f}<br>Run: " + run + "<extra></extra>",
                        ),
                        row=i,
                        col=1,
                    )

            # Add median scatter for the current experiment
            median_key = f"median_{measure}"
            if median_key in results_per_run[current_exp]:
                median_values = results_per_run[current_exp][median_key]['values']
                query_ids = results_per_run[current_exp][median_key]['query_ids']
                fig.add_trace(
                    go.Scatter(
                        x=query_ids,
                        y=median_values,
                        mode='markers',
                        name=f'Median (N-1)',
                        marker=dict(
                            color='red',
                            symbol='star',
                            size=10,
                            line=dict(width=0.5, color='DarkSlateGrey')
                        ),
                        showlegend=(i == 1),
                        hovertemplate="Query: %{x}<br>Median (N-1): %{y:.3f}<extra></extra>",
                    ),
                    row=i,
                    col=1,
                )

            # Update y-axis title
            fig.update_yaxes(title_text=measure, row=i, col=1)

            # Update x-axis settings
            fig.update_xaxes(
                title_text="Query ID",
                tickmode="array",
                tickvals=query_ids,
                ticktext=query_ids,
                tickangle=45,
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=550 * len(eval_measures),  # Increase height
            title={
                "text": f"""Performance of <span style="color:red;">{current_exp}</span> compared to the Median performance calculated based on the remaining <span style="color:red;">{len(runs)-1}</span> experiments.""",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=100, b=50),
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_performance_difference_threshold(data, threshold):
    # Extract measures and runs
    eval_measures = list(data[list(data.keys())[0]].keys())
    runs = list(data.keys())

    # Generate colors
    colors = generate_colors(len(runs))

    # Create a subplot for each measure
    fig = make_subplots(
        rows=len(eval_measures),
        cols=1,
        subplot_titles=eval_measures,
        vertical_spacing=0.15,
    )

    for i, measure in enumerate(eval_measures, start=1):
        for j, run in enumerate(runs):
            values = data[run][measure][measure]['values']
            query_ids = data[run][measure][measure]['query_ids']

            # Calculate the difference from the threshold
            diff_values = [val - threshold for val in values]

            fig.add_trace(
                go.Bar(
                    x=query_ids,
                    y=diff_values,
                    name=run,
                    marker_color=colors[j],
                    showlegend=(i == 1),  # Only show legend for the first subplot
                    hovertemplate="Query: %{x}<br>Difference: %{y:.3f}<br>Run: "
                    + run
                    + "<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=query_ids[0],
            x1=query_ids[-1],
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
            row=i,
            col=1,
        )

        # Update y-axis title
        fig.update_yaxes(title_text=f"{measure} Difference from Threshold", row=i, col=1)

        # Update x-axis settings
        fig.update_xaxes(
            title_text="Query ID",
            tickmode="array",
            tickvals=query_ids,  # Show every 5th tick
            ticktext=query_ids,
            tickangle=45,  # Rotate labels 90 degrees
            row=i,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=550 * len(eval_measures),  # Increase height
        title={
            "text": f"""Performance Difference Compared to Threshold (<span style="color:red;">{threshold:.2f}</span>)""",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_query_relevance_judgements(selected_qrel):
    relevance_counts, results = get_query_rel_judgements(selected_qrel)

    # Sort the query IDs
    sorted_query_ids = sort_query_ids(relevance_counts.index)
    relevance_counts = relevance_counts.loc[sorted_query_ids]

    # Create subplots
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=["Relevance Judgements per Query"]
    )

    # Create color scale for relevance labels
    relevant_columns = [col for col in relevance_counts.columns if col != "Irrelevant"]
    num_relevant_labels = len(relevant_columns)

    if num_relevant_labels > 1:
        blue_scale = mcolors.LinearSegmentedColormap.from_list(
            "", ["lightblue", "darkblue"]
        )
        blue_colors = [
            mcolors.rgb2hex(blue_scale(i / (num_relevant_labels - 1)))
            for i in range(num_relevant_labels)
        ]
    elif num_relevant_labels == 1:
        blue_colors = ["blue"]
    else:
        blue_colors = []

    # Create traces for each relevance level
    for i, column in enumerate(relevance_counts.columns):
        if column == "Irrelevant":
            color = "red"
        else:
            color = blue_colors[relevant_columns.index(column)]

        fig.add_trace(
            go.Bar(
                name=column,
                x=relevance_counts.index,
                y=relevance_counts[column],
                text=relevance_counts[column],
                textposition="auto",
                marker_color=color,
                hovertemplate="Query: %{x}<br>" + f"{column}: " + "%{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=550,
        title={
            "text": "Query Relevance Judgements Analysis",
            "x": 0.01,
            "xanchor": "left",
            "yanchor": "top",
        },
        barmode="stack",
        xaxis_title="Query ID",
        yaxis_title="Number of Documents",
        legend_title="Relevance Labels",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update x-axis settings
    fig.update_xaxes(
        tickmode="array",
        tickvals=relevance_counts.index,
        ticktext=relevance_counts.index,
        tickangle=45,
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    return results


@st.cache_data
def plot_query_performance_vs_query_length_moving_avg(df, measure):
    # Sort DataFrame by Token Length
    df = df.sort_values("Token Length")

    # Calculate moving average
    window = 20  # Adjust this value to change the smoothness of the line
    df["MA"] = df["Performance"].rolling(window=window, center=True).mean()

    # Create figure
    fig = go.Figure()

    # Add scatter plot for individual queries
    fig.add_trace(
        go.Scatter(
            x=df["Token Length"],
            y=df["Performance"],
            mode="markers",
            name="Individual Queries",
            marker=dict(
                size=6,
                color=df["Performance"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=measure),
            ),
            text=[
                f"Query ID: {qid}<br>Token Length: {tl}<br>{measure}: {perf:.4f}"
                for qid, tl, perf in zip(
                    df["Query ID"], df["Token Length"], df["Performance"]
                )
            ],
            hoverinfo="text",
        )
    )

    # Add moving average line
    fig.add_trace(
        go.Scatter(
            x=df["Token Length"],
            y=df["MA"],
            mode="lines",
            name=f"Moving Average (window={window})",
            line=dict(color="red", width=2),
        )
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Token Length",
        title={
            "text": f"""<span style="color:red;">{measure}</span> vs Query Length""",
            "x": 0.01,
            "xanchor": "left",
            "yanchor": "top",
        },
        yaxis_title=measure,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@st.cache_data
def plot_query_performance_vs_query_length_buckets(df, measure):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Equal-width Buckets", "Equal-frequency Buckets"],
    )

    # Function to create hover text
    def create_hover_text(row):
        return (
            f"Bucket Range: {row.name}<br>"
            f"Mean {measure}: {row['Performance']['mean']:.4f}<br>"
            f"Number of Queries: {row['Performance']['count']}<br>"
            f"Queries: {', '.join(map(str, row['Query ID']))}<br>"
            f"Token Lengths: {', '.join(map(str, row['Token Length']))}"
        )

    # Equal-width buckets
    df["Token Length Bucket"] = pd.cut(df["Token Length"], bins=10)
    grouped_equal_width = df.groupby("Token Length Bucket").agg(
        {"Performance": ["mean", "count"], "Query ID": list, "Token Length": list}
    )

    x_labels_equal_width = [
        f"{int(interval.left)}-{int(interval.right)}"
        for interval in grouped_equal_width.index
    ]
    hover_text_equal_width = grouped_equal_width.apply(create_hover_text, axis=1)

    fig.add_trace(
        go.Bar(
            x=x_labels_equal_width,
            y=grouped_equal_width["Performance"]["mean"],
            name="Equal-width",
            hovertext=hover_text_equal_width,
            hoverinfo="text",
            text=grouped_equal_width["Performance"]["count"],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # Equal-frequency buckets
    df["Token Length Bucket"] = pd.qcut(df["Token Length"], q=10, duplicates="drop")
    grouped_equal_freq = df.groupby("Token Length Bucket").agg(
        {"Performance": ["mean", "count"], "Query ID": list, "Token Length": list}
    )

    x_labels_equal_freq = [
        f"{int(interval.left)}-{int(interval.right)}"
        for interval in grouped_equal_freq.index
    ]
    hover_text_equal_freq = grouped_equal_freq.apply(create_hover_text, axis=1)

    fig.add_trace(
        go.Bar(
            x=x_labels_equal_freq,
            y=grouped_equal_freq["Performance"]["mean"],
            name="Equal-frequency",
            hovertext=hover_text_equal_freq,
            hoverinfo="text",
            text=grouped_equal_freq["Performance"]["count"],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Token Length Ranges", tickangle=45)
    fig.update_yaxes(title_text=f"Mean {measure}")

    fig.update_layout(
        height=450,
        showlegend=False,
        title={
            "text": f"""Mean <span style="color:red;">{measure}</span> vs Query Length Buckets""",
            "x": 0.01,
            "xanchor": "left",
            "yanchor": "top",
        },
    )
    return fig


@st.cache_data
def plot_query_performance_vs_query_length(data):
    # Debug: Print the structure of the results
    # st.write("Debug: Results structure")
    # st.write(data)
    #
    # # Flatten the nested dictionary structure
    # flattened_data = []
    # for run, run_data in data.items():
    #     for measure, measure_data in run_data.items():
    #         if measure != 'token_length':
    #             for query_id, score in zip(measure_data['query_id'], measure_data[measure]):
    #                 flattened_data.append({
    #                     'Run': run,
    #                     'Measure': measure,
    #                     'Query ID': query_id,
    #                     'Score': score,
    #                     'Token Length': run_data['token_length']['token_length'].get(str(query_id), -1)
    #                 })

    experiments = list(data.keys())
    measures_eval = [
        measure for measure in data[experiments[0]].keys() if measure != "token_length"
    ]

    for experiment in experiments:
        st.write(
            f"""<center><h5>Analysis of the <span style="color:red;">{experiment}</span> Experiment</h5></center>""",
            unsafe_allow_html=True,
        )

        for measure in measures_eval:
            # Create DataFrame with token lengths, performance, and query IDs
            df = pd.DataFrame([
                {
                    "Token Length": data[experiment]['token_length']['token_length'].get(str(query_id), -1),
                    "Performance": score,
                    "Query ID": query_id
                }
                for query_id, score in zip(data[experiment][measure]['query_id'], data[experiment][measure][measure])
            ])

            # Remove rows where Token Length is -1 (placeholder for missing data)
            df = df[df["Token Length"] != -1]

            if df.empty:
                st.warning(f"No valid data available for {measure} in {experiment}. All token lengths were missing.")
                continue

            # Create and display the combined plot
            fig_combined = plot_query_performance_vs_query_length_moving_avg(
                df, measure
            )
            st.plotly_chart(fig_combined, use_container_width=True)

            # Bucket analysis
            fig_buckets = plot_query_performance_vs_query_length_buckets(df, measure)
            st.plotly_chart(fig_buckets, use_container_width=True)

            # ANOVA analysis
            if len(df) >= 20:  # Ensure we have enough data points for meaningful analysis
                f_statistic_ew, p_value_ew = stats.f_oneway(
                    *[
                        group["Performance"].values
                        for name, group in df.groupby(
                            pd.cut(df["Token Length"], bins=10), observed=False
                        )
                        if len(group) > 0  # Ensure each group has at least one value
                    ]
                )
                f_statistic_ef, p_value_ef = stats.f_oneway(
                    *[
                        group["Performance"].values
                        for name, group in df.groupby(
                            pd.qcut(df["Token Length"], q=10, duplicates="drop"),
                            observed=False,
                        )
                        if len(group) > 0  # Ensure each group has at least one value
                    ]
                )

                if p_value_ew < 0.05 or p_value_ef < 0.05:
                    st.write(
                        f"""At least one of the bucketing methods shows statistically significant differences between buckets (p < <span style="color: red;">0.05</span>). 
                                ANOVA results (Equal-width buckets): F-statistic = <span style="color: red;">{f_statistic_ew:.4f}</span>, p-value = <span style="color: red;">{p_value_ew:.4f}.</span>
                                ANOVA results (Equal-frequency buckets): F-statistic = <span style="color: red;">{f_statistic_ef:.4f}</span>, p-value = <span style="color: red;">{p_value_ef:.4f}.</span>
                                """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(
                        f"""Neither bucketing method shows statistically significant differences between buckets (p >= <span style="color: red;">0.05</span>).
                                ANOVA results (Equal-width buckets): F-statistic = <span style="color: red;">{f_statistic_ew:.4f}</span>, p-value = <span style="color: red;">{p_value_ew:.4f}.</span>
                                ANOVA results (Equal-frequency buckets): F-statistic = <span style="color: red;">{f_statistic_ef:.4f}</span>, p-value = <span style="color: red;">
                                {p_value_ef:.4f}.</span>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning(f"Not enough data points for ANOVA analysis in {measure} for {experiment}.")


@st.cache_resource
def create_relevance_wordclouds(query_relevance, queries, method, threshold) -> None:
    # Ensure query_ids in the DataFrame are strings
    queries["query_id"] = queries["query_id"].astype(str)

    q_ids_classified = query_clf_relevance_assessments(
        query_relevance, method, threshold
    )

    # Convert all query IDs in q_ids_classified to strings
    q_ids_classified = {
        relevance_class: {
            subclass: [str(qid) for qid in query_ids]
            for subclass, query_ids in subclasses.items()
        }
        for relevance_class, subclasses in q_ids_classified.items()
    }

    def create_wordcloud(text):
        if not text.strip():
            return None
        try:
            return WordCloud(
                width=400,
                height=200,
                background_color="white",
                max_words=50,
                min_font_size=10,
                collocations=False,
                min_word_length=3,
            ).generate(text)
        except ValueError:
            return None

    # Add an expander with explanation
    with st.expander("Explanation of Word Clouds and Term Calculation"):
        st.write(
            """
        This visualization presents word clouds created based on sampled queries (using the aforementioned methods) for the different relevance labels. 

        1. Word Clouds: Each word cloud represents the most frequent words appearing in the sampled queries. 
           The size of each word corresponds to its frequency - larger words appear more often in the queries.

        4. Query Information: We show the IDs of the queries that contributed to each word cloud or single query display.

        Calculation Process:
        - All queries are classified into each subclass.
        - For multiple queries, stopwords (common words like 'the', 'a', 'an'), numbers, and punctuation are removed.
        - All remaining words from these queries.
        - Word frequency is calculated from this combined text.
        - The word cloud is generated based on these frequencies.
        """
        )

    for relevance_class, subclasses in q_ids_classified.items():
        # Determine color based on relevance class
        class_color = "red" if relevance_class.lower() == "irrelevant" else "blue"
        relevance_class_name = (
            "Relevance_Label_0 (Irrelevant)"
            if relevance_class.lower() == "irrelevant"
            else relevance_class
        )
        st.markdown(
            f"""<h5>Analysis based on the <span style="color:{class_color};">{relevance_class_name}</span></h5>""",
            unsafe_allow_html=True,
        )

        # Create a 3x2 grid of columns
        cols = st.columns(3)

        for i, (subclass, query_ids) in enumerate(subclasses.items()):
            with cols[i % 3]:
                if len(query_ids) == 0:
                    # Case: No queries
                    st.write(
                        """No queries classified as this subclass.""",
                        unsafe_allow_html=True,
                    )
                elif len(query_ids) == 1:
                    # Case: Single query
                    query = queries[queries["query_id"] == query_ids[0]]
                    if not query.empty:
                        st.write(
                            f"""This class that contains <span style="color:red;">{str(subclass).replace('_', ' ')}</span> consisted of one query with ID {query_ids[0]}.""",
                            unsafe_allow_html=True,
                        )
                        with st.expander("See query text"):
                            st.write(query["query_text"].iloc[0])
                    else:
                        st.write(
                            f"Query with ID {query_ids[0]} not found in the dataset."
                        )
                else:
                    # Case: Multiple queries
                    relevant_queries = queries[queries["query_id"].isin(query_ids)]

                    if not relevant_queries.empty:
                        relevant_queries_no_stopwords = remove_stopwords_from_queries(
                            relevant_queries
                        )
                        all_text = " ".join(relevant_queries_no_stopwords["query_text"])

                        wordcloud = create_wordcloud(all_text)

                        if wordcloud:
                            fig, ax = plt.subplots(figsize=(5, 2.5))
                            ax.imshow(wordcloud, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)
                        else:
                            st.write(
                                "Could not generate word cloud due to insufficient unique words."
                            )

                        # Create message based on subclass
                        if subclass == "queries_above_95th_percentile":
                            message = f"""Wordcloud created based on the queries (<span style="color:red;">{', '.join(query_ids)}</span>) that have relevance judgements **above the 95th 
                            percentile.**"""
                        elif subclass == "queries_between_5th_95th_percentiles":
                            message = f"""Wordcloud created based on the queries (<span style="color:red;">{', '.join(query_ids)}</span>) that have relevance judgements **between the 5th and 95th percentiles.**"""
                        elif subclass == "queries_below_5th_percentile":
                            message = f"""Wordcloud created based on the queries (<span style="color:red;">{', '.join(query_ids)}</span>) that have relevance judgements **below the 5th percentile.**"""
                        elif subclass == "5_queries_most_assessments":
                            message = f"""Wordcloud created based on the queries (<span style="color:red;">{', '.join(query_ids)}</span>) that have the **most relevance judgements.**"""
                        elif subclass == "5_queries_around_median_assessments":
                            message = f"""Wordcloud created based on the queries (<span style="color:red;">{', '.join(query_ids)}</span>) that have relevance judgements **close to the median 
                            relevance judgements** of all analyzed queries."""
                        elif subclass == "5_queries_least_assessments":
                            message = f"""Wordcloud created based on the queries (<span style="color:red;">{', '.join(query_ids)}</span>) that have the **least relevance judgements.**"""
                        elif subclass == "queries_above_threshold":
                            message = f"""Wordcloud created based on the queries (<span style="color:red;">{', '.join(query_ids)}</span>) that are **above the selected threshold.**"""
                        elif subclass == "queries_below_threshold":
                            message = f"""Wordcloud created based on the queries (<span style="color:red;">{', '.join(query_ids)}</span>) that are **below the selected threshold.**"""
                        elif subclass == "queries_within_normal_range":
                            message = f"""Wordcloud created based on the queries (<span style="color:red;">{', '.join(query_ids)}</span>) that have relevance judgements **close to threshold.**"""
                        else:
                            message = f"""The wordcloud is generated based on the text of the following queries: <span style="color:red;">{', '.join(query_ids)}</span>"""

                        # Add query information below the figure
                        st.write(f"""{message}""", unsafe_allow_html=True)
                    else:
                        st.write("No matching queries found in the dataset.")


@st.cache_data
def plot_performance_similarity(
    queries,
    qrel,
    runs,
    metric_list,
    selected_cutoff,
    relevance_threshold,
    embedding_model_name,
) -> None:
    pca_df, results_per_run = query_similarity_performance(
        queries,
        qrel,
        runs,
        metric_list,
        selected_cutoff,
        relevance_threshold,
        embedding_model_name,
    )

    for experiment, measures in results_per_run.items():
        st.write(
            f"""<center><h5>Analysis of the <span style="color:red;">{experiment}</span> Experiment</h5></center>""",
            unsafe_allow_html=True,
        )

        eval_measures = [
            measure for measure in measures.keys() if measure != "token_length"
        ]

        for measure in eval_measures:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=[
                    f"2D {measure} Performance",
                    f"3D {measure} Performance",
                ],
                specs=[[{"type": "xy"}, {"type": "scene"}]],
            )

            # 2D Plot
            scatter_2d = go.Scatter(
                x=pca_df["Prin. Comp. 1"],
                y=pca_df["Prin. Comp. 2"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=measures[measure][measure],
                    colorscale="RdYlBu_r",
                    showscale=False,
                    colorbar=dict(
                        title=measure,
                        thickness=15,
                        len=0.9,
                        yanchor="middle",
                        y=0.5,
                        x=0.45,
                        outlinewidth=0,
                    ),
                ),
                showlegend=False,
                text=[
                    f"Query ID: {q_id}<br>{measure}: {score:.4f}"
                    for q_id, score in zip(
                        pca_df["query_id"], measures[measure][measure]
                    )
                ],
                hoverinfo="text",
            )

            # 3D Plot
            scatter_3d = go.Scatter3d(
                x=pca_df["Prin. Comp. 1"],
                y=pca_df["Prin. Comp. 2"],
                z=pca_df["Prin. Comp. 3"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=measures[measure][measure],
                    colorscale="RdYlBu_r",
                    showscale=True,
                    colorbar=dict(
                        title=measure,
                        thickness=15,
                        len=0.9,
                        yanchor="middle",
                        y=0.5,
                        x=1.0,
                        outlinewidth=0,
                    ),
                ),
                showlegend=False,
                text=[
                    f"Query ID: {q_id}<br>{measure}: {score:.4f}"
                    for q_id, score in zip(
                        pca_df["query_id"], measures[measure][measure]
                    )
                ],
                hoverinfo="text",
            )

            fig.add_trace(scatter_2d, row=1, col=1)
            fig.add_trace(scatter_3d, row=1, col=2)

            fig.update_layout(
                height=600,
                width=1200,
                title_text=f"{measure} Performance - 2D and 3D Visualization",
                scene=dict(
                    xaxis_title="Prin. Comp. 1",
                    yaxis_title="Prin. Comp. 2",
                    zaxis_title="Prin. Comp. 3",
                ),
            )

            fig.update_xaxes(title_text="Prin. Comp. 1", row=1, col=1)
            fig.update_yaxes(title_text="Prin. Comp. 2", row=1, col=1)

            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Manually Examine Sampled Queries"):
            st.dataframe(
                queries[["query_id", "query_text"]],
                use_container_width=True,
                hide_index=True,
            )


@st.cache_data
def plot_multi_query_docs(multi_query_docs):
    fig = go.Figure()

    # Get unique relevance labels
    all_relevance = [
        rel
        for doc_rel in multi_query_docs["relevance_judgments"]
        for rel in doc_rel.values()
    ]
    unique_relevance = sorted(set(all_relevance))

    # Generate a color scale
    color_scale = px.colors.qualitative.Plotly
    relevance_colors = {
        rel: color_scale[i % len(color_scale)] for i, rel in enumerate(unique_relevance)
    }

    for rel_label in unique_relevance:
        counts = [
            sum(1 for rel in doc_rel.values() if rel == rel_label)
            for doc_rel in multi_query_docs["relevance_judgments"]
        ]

        fig.add_trace(
            go.Bar(
                x=multi_query_docs["doc_id"],
                y=counts,
                name=f"Relevance {rel_label}",
                marker_color=relevance_colors[rel_label],
                hovertext=[
                    f"Queries: {', '.join([q for q, r in doc_rel.items() if r == rel_label])}"
                    for doc_rel in multi_query_docs["relevance_judgments"]
                ],
                hoverinfo="text+y",
                hoverlabel=dict(
                    bgcolor='white',
                    font=dict(color='gray')
                ),
            )
        )

    fig.update_layout(
        title="Documents' Relevance Judged across Multiple Queries",
        xaxis_title="Document ID",
        yaxis_title="Number of Queries",
        barmode="stack",
        height=600,
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend_title="Relevance Label",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_documents_retrieved_by_experiments(result_df, excluded_runs=None) -> None:
    """
    Analyze and display results of documents retrieved by different numbers of experiments.

    :param result_df: DataFrame containing the aggregated results
    :param excluded_runs: List of run names to exclude from the analysis
    """
    if excluded_runs:
        # Filter out excluded runs
        result_df["run"] = result_df["run"].apply(
            lambda x: ",".join(
                [run for run in x.split(",") if run not in excluded_runs]
            )
        )
        result_df["occurrence_count"] = result_df["run"].apply(
            lambda x: len(x.split(",")) if x else 0
        )
        result_df = result_df[result_df["occurrence_count"] > 0]

    total_experiments = len(set(",".join(result_df["run"]).split(",")))
    total_pairs = len(result_df)

    st.write(
        f"Number of experiments included in the analysis: <span style='color:red;'>{total_experiments}</span>",
        unsafe_allow_html=True,
    )
    st.write(
        f"Total unique query-document pairs retrieved across all experiments: <span style='color:red;'>{total_pairs}</span>",
        unsafe_allow_html=True,
    )

    # Create separate columns for different retrieval thresholds
    thresholds = [1, 2, 3, 5]
    threshold_data = []
    for threshold in thresholds:
        if threshold <= total_experiments:
            result_df[f"by_{threshold}_exps"] = (
                result_df["occurrence_count"] == threshold
            )
            count = result_df[f"by_{threshold}_exps"].sum()
            percentage = count / total_pairs
            st.write(
                f"<h9><b>Query-document pairs retrieved by exactly {threshold} experiment{'s' if threshold > 1 else ''} and no others: "
                f"<span style='color:red;'>{count}</span> (<span style='color:red;'>{percentage:.2%}</span> of total unique pairs)</b></h9>",
                unsafe_allow_html=True,
            )
            threshold_data.append(
                {
                    "Threshold": f"{threshold} exp{'s' if threshold > 1 else ''}",
                    "Count": count,
                    "Percentage": percentage,
                }
            )

            # Display example pairs for all thresholds
            filtered_df = result_df[result_df[f"by_{threshold}_exps"]]
            if len(filtered_df) >= 10:
                example_pairs = filtered_df.sample(10)
            else:
                # Handle the case where fewer than 10 rows are present
                example_pairs = filtered_df

            with st.expander(
                f"Example pairs retrieved by exactly {threshold} experiment{'s' if threshold > 1 else ''}"
            ):
                for _, row in example_pairs.iterrows():
                    st.write(
                        f"Query ID: <span style='color:red;'>{row['query_id']}</span>, Document ID: <span style='color:red;'>{row['doc_id']}</span>, Retrieved by: <span style='color:red;'>"
                        f"{row['run']}</span>",
                        unsafe_allow_html=True,
                    )

    # Special cases
    result_df["by_all_exps"] = result_df["occurrence_count"] == total_experiments
    half_plus_one_threshold = total_experiments // 2 + 1
    result_df["by_half_plus_one_exps"] = (
        result_df["occurrence_count"] >= half_plus_one_threshold
    )

    half_plus_one = result_df["by_half_plus_one_exps"].sum()
    half_plus_one_percentage = half_plus_one / total_pairs
    st.write(
        f"<h9><b>Query-document pairs retrieved by at least half+1 of the experiments ({half_plus_one_threshold}): "
        f"<span style='color:red;'>{half_plus_one}</span> (<span style='color:red;'>{half_plus_one_percentage:.2%}</span> of total unique pairs)</b></h9>",
        unsafe_allow_html=True,
    )
    threshold_data.append(
        {
            "Threshold": "Half+1 exps",
            "Count": half_plus_one,
            "Percentage": half_plus_one_percentage,
        }
    )

    # Display example pairs for half+1
    half_plus_one_pairs = result_df[result_df["by_half_plus_one_exps"]].sample(10)
    with st.expander(
        f"Example pairs retrieved by at least {half_plus_one_threshold} experiments (half+1)"
    ):
        for _, row in half_plus_one_pairs.iterrows():
            st.write(
                f"Query ID: <span style='color:red;'>{row['query_id']}</span>, Document ID: <span style='color:red;'>{row['doc_id']}</span>, Retrieved by: <span style='color:red;'>"
                f"{row['run']}</span>",
                unsafe_allow_html=True,
            )

    all_exps = result_df["by_all_exps"].sum()
    all_exps_percentage = all_exps / total_pairs
    st.write(
        f"<h9><b>Query-document pairs retrieved by all {total_experiments} experiments: "
        f"<span style='color:red;'>{all_exps}</span> (<span style='color:red;'>{all_exps_percentage:.2%}</span> of total unique pairs)</b></h9>",
        unsafe_allow_html=True,
    )
    threshold_data.append(
        {"Threshold": "All exps", "Count": all_exps, "Percentage": all_exps_percentage}
    )

    # Display example pairs for all experiments
    all_exps_pairs = result_df[result_df["by_all_exps"]].sample(10)
    with st.expander(f"Example pairs retrieved by all {total_experiments} experiments"):
        for _, row in all_exps_pairs.iterrows():
            st.write(
                f"Query ID: <span style='color:red;'>{row['query_id']}</span>, Document ID: <span style='color:red;'>{row['doc_id']}</span>, Retrieved by: <span style='color:red;'>"
                f"{row['run']}</span>",
                unsafe_allow_html=True,
            )

    st.write(
        "<h5>Query Difficulty Analysis based on uniquely retrieved Documents</h5>",
        unsafe_allow_html=True,
    )

    # Query difficulty analysis
    with st.expander("See Analysis"):
        st.write(
            """
        This section analyzes the difficulty of each query based on the retrieval results. 
        We define query difficulty as the proportion of documents for that query that were retrieved by only one experiment.

        - A higher percentage indicates a more difficult query (more unique retrievals).
        - A lower percentage indicates an easier query (more agreement between experiments).
           
        Interpretation guide:
           - Red colors indicate more difficult queries (higher percentage of unique retrievals).
           - Green colors indicate easier queries (lower percentage of unique retrievals).
           - The percentage shows how many of the documents for each query were retrieved by only one experiment.

        Calculation method:
        1. For each query, we count the number of documents retrieved by only one experiment.
        2. We divide this count by the total number of documents retrieved for that query.
        3. The result is expressed as a percentage.

        For example:
        Let's say for Query A, we have the following retrieval results:
        - Document 1: Retrieved by Experiment 1, 2, and 3
        - Document 2: Retrieved by Experiment 1 only
        - Document 3: Retrieved by Experiment 2 and 3
        - Document 4: Retrieved by Experiment 1 only
        - Document 5: Retrieved by all experiments

        In this case:
        - Total documents retrieved: 5
        - Documents retrieved by only one experiment: 2 (Documents 2 and 4)
        - Difficulty score = 2 / 5 = 0.4 or 40%

        This means that 40% of the documents retrieved for Query A were unique to a single experiment, 
        suggesting a moderate level of difficulty.

        Below, you'll see all queries ranked from most difficult to easiest based on this calculation.
        """
        )

        query_difficulty = result_df.groupby("query_id").apply(
            lambda x: (x["occurrence_count"] == 1).sum() / len(x)
        )
        query_difficulty_sorted = query_difficulty.sort_values(ascending=False)

        st.write(
            "<h6>All Queries Ranked by Difficulty (Hardest to Easiest)</h6>",
            unsafe_allow_html=True,
        )

        # Create a dataframe for display
        difficulty_df = pd.DataFrame(
            {
                "Query ID": query_difficulty_sorted.index,
                "Difficulty Score": query_difficulty_sorted.values,
            }
        )

        # Color formatting function
        def color_difficulty(val):
            color = f"rgb({int(255 * val)}, {int(255 * (1 - val))}, 0)"
            return f"color: {color}"

        cola, colb = st.columns(2)
        with cola:
            # Apply color formatting and display
            st.dataframe(
                difficulty_df.style.format({"Difficulty Score": "{:.2%}"}).map(
                    color_difficulty, subset=["Difficulty Score"]
                ),
                hide_index=True,
                use_container_width=True,
            )
        with colb:
            st.write(
                """
               What does this mean?
               - Queries at the top of the list (with higher percentages) are more difficult. These queries have a higher proportion of documents that were only retrieved by one experiment.
               - Queries at the bottom of the list (with lower percentages) are easier. These queries have more agreement between different retrieval methods.
               - The distribution of difficulty can help you understand if certain types of queries are consistently challenging across different retrieval methods.
               - If you see a clear divide between difficult and easy queries, it might indicate distinct categories of queries in your dataset.
               """
            )

            st.write(
                """
               Possible Next steps:
               - For difficult queries (those with high percentages), you might want to examine the specific documents that were uniquely retrieved.
               - For easy queries (those with low percentages), you could look at which documents were consistently retrieved across experiments.
               """
            )

    # Display top documents for a specific query
    st.write(
        "<h5>Identify Commonly and Uniquely retrieved Documents per Query</h5>",
        unsafe_allow_html=True,
    )
    with st.expander("See Analysis"):
        query_ids = sorted(result_df["query_id"].unique())
        selected_query = st.selectbox("Select a query to view documents:", query_ids)

        if selected_query:
            st.write(
                f"Document retrieval analysis for query <span style='color:red;'>{selected_query}</span>.",
                unsafe_allow_html=True,
            )

            query_docs = result_df[result_df["query_id"] == selected_query]
            total_experiments = len(set(",".join(query_docs["run"]).split(",")))

            col1, col2 = st.columns(2)

            with col1:
                st.write("Documents retrieved by all systems:")
                all_systems_docs = query_docs[
                    query_docs["occurrence_count"] == total_experiments
                ]
                if not all_systems_docs.empty:
                    st.dataframe(
                        all_systems_docs[["doc_id", "occurrence_count", "run"]],
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.write(
                        "No documents were retrieved by all systems for this query."
                    )

            with col2:
                single_system_docs = query_docs[query_docs["occurrence_count"] == 1]
                two_system_docs = pd.DataFrame()  # Initialize as empty DataFrame
                if not single_system_docs.empty:
                    st.write("Documents retrieved by only one system:")
                    st.dataframe(
                        single_system_docs[["doc_id", "occurrence_count", "run"]],
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    two_system_docs = query_docs[query_docs["occurrence_count"] == 2]
                    if not two_system_docs.empty:
                        st.write(
                            "No documents were retrieved by only one system. Showing documents retrieved by two systems:"
                        )
                        st.dataframe(
                            two_system_docs[["doc_id", "occurrence_count", "run"]],
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.write(
                            "No documents were retrieved by only one or two systems for this query."
                        )

            # Calculate and display percentages
            total_docs = len(query_docs)
            all_systems_percentage = len(all_systems_docs) / total_docs * 100
            single_system_percentage = len(single_system_docs) / total_docs * 100
            two_system_percentage = len(two_system_docs) / total_docs * 100

            with col1:
                st.write(
                    f"- Total unique documents retrieved for this query: <span style='color:red;'>{total_docs}</span>",
                    unsafe_allow_html=True,
                )
                st.write(
                    f"- Percentage of documents retrieved by all systems: <span style='color:red;'>{all_systems_percentage:.2f}%</span>",
                    unsafe_allow_html=True,
                )
                st.write(
                    f"- Percentage of documents retrieved by only one system: <span style='color:red;'>{single_system_percentage:.2f}%</span>",
                    unsafe_allow_html=True,
                )
                if single_system_percentage == 0:
                    st.write(
                        f"- Percentage of documents retrieved by only two systems: <span style='color:red;'>{two_system_percentage:.2f}%</span>",
                        unsafe_allow_html=True,
                    )

            with col2:
                st.write(
                    """
                Interpretation:
                - Documents retrieved by all systems represent a strong consensus among retrieval methods.
                - Documents retrieved by only one (or two) systems might be unique findings or potential noise.
                - A high percentage of all-system retrievals suggests good agreement among methods for this query.
                - A high percentage of single-system retrievals might indicate a challenging or ambiguous query.
                """
                )


@st.cache_data
def plot_rankings_docs_rel_ids(qrel, runs, ranking_depth):
    def generate_color_scale(num_colors):
        colors = ["lightgray"]  # For unjudged (-100)
        if num_colors > 1:
            # Create a custom colormap from red to green
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom", ["red", "orange", "green"]
            )
            colors.extend(
                [mcolors.rgb2hex(cmap(i / (num_colors - 1))) for i in range(num_colors)]
            )
        else:
            colors.append("red")  # If there's only one relevance level
        return colors

    # Determine unique relevance values and generate color scale
    unique_relevance = sorted(qrel["relevance"].unique())
    color_scale = generate_color_scale(len(unique_relevance))

    # Create normalized colorscale for Plotly
    min_val, max_val = -100, max(unique_relevance)
    range_val = max_val - min_val
    colorscale = [
        [(val - min_val) / range_val, color]
        for val, color in zip([-100] + unique_relevance, color_scale)
    ]

    # Determine grid layout
    num_runs = len(runs)
    num_cols = 2
    num_rows = math.ceil(num_runs / num_cols)

    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[
            get_experiment_name(run_name, None) for run_name in runs.keys()
        ],
        # vertical_spacing=0.15,
        horizontal_spacing=0.05,
    )

    for i, (run_name, run_df) in enumerate(runs.items(), start=1):
        # Merge run data with qrel data to get relevance information
        merged_df = pd.merge(run_df, qrel, on=["query_id", "doc_id"], how="left")
        merged_df["relevance"] = merged_df["relevance"].fillna(
            -100
        )  # Mark unjudged as -100

        # Filter for the specified ranking depth
        merged_df = merged_df[merged_df["rank"] <= ranking_depth]

        # Create heatmap data and hover text
        heatmap_data = []
        hover_text = []

        for rank in range(1, ranking_depth + 1):
            rank_data = merged_df[merged_df["rank"] == rank]
            row_data = rank_data["relevance"].tolist()
            row_data += [None] * (
                len(merged_df["query_id"].unique()) - len(row_data)
            )  # Pad with None
            heatmap_data.append(row_data)

            hover_row = [
                f"Query: {query_id}<br>Rank: {rank}<br>Doc ID: {doc_id}<br>Relevance: {'Unjudged' if rel == -100 else rel}"
                for query_id, doc_id, rel in zip(
                    rank_data["query_id"], rank_data["doc_id"], rank_data["relevance"]
                )
            ]
            hover_row += [None] * (len(merged_df["query_id"].unique()) - len(hover_row))
            hover_text.append(hover_row)

        row = (i - 1) // num_cols + 1
        col = (i - 1) % num_cols + 1

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                y=list(range(1, ranking_depth + 1)),
                x=merged_df["query_id"].unique(),
                colorscale=colorscale,
                showscale=False,
                hoverinfo="text",
                text=hover_text,
                xgap=1,
                ygap=1,
                zmin=min_val,
                zmax=max_val,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text="Query ID", row=row, col=col, tickangle=45)
        fig.update_yaxes(
            title_text="Ranking Position" if col == 1 else "", row=row, col=col
        )

        # Add legend only once
        if i == 1:
            for val, color in zip([-100] + unique_relevance, color_scale):
                label = (
                    "Unjudged Relevance" if val == -100 else f"Relevance_Label_{val}"
                )
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(size=10, color=color),
                        name=label,
                        legendgroup=f"relevance {val}",
                        showlegend=True,
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        height=700 * num_rows,
        width=1400,
        title_text=f"Document Ranking and Relevance for the Top {ranking_depth} Rank Positions per Experiment",
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)
