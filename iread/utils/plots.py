import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
import plotly.graph_objects as go
from utils.eval_single_exp import *
from plotly.subplots import make_subplots
import colorsys


# Function that displays the distribution of ranking position of all retrieved documents based on their relevance label.
@st.cache_resource
def dist_of_retrieved_docs(relevance_ret_pos: dict) -> None:
    # Define constants for bucket ranges
    bucket_ranges = {
        '1': (1, 1),
        '2-10': (2, 10),
        '11-20': (11, 20),
        '21-30': (21, 30),
        '31-40': (31, 40),
        '41-50': (41, 50),
        '51-60': (51, 60),
        '61-70': (61, 70),
        '71-80': (71, 80),
        '81-90': (81, 90),
        '91-100': (91, 100),
        '101-200': (101, 200),
        '200+': (201, 999)  # Handle values greater than 200
    }

    # Initialize buckets dictionary dynamically
    thresholds = set(key for values in relevance_ret_pos.values() for key in values.keys())
    buckets = {threshold: {bucket: 0 for bucket in bucket_ranges.keys()} for threshold in thresholds}

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
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold']  # Different colors for each metric
    width = 0.2  # Width of each bar

    for index, (metric, bucket_counts) in enumerate(sorted_buckets.items()):
        fig.add_trace(go.Bar(
            x=[i + (index - num_metrics / 2) * width for i in x_indices],
            y=[bucket_counts[bucket] for bucket in x_labels],
            width=width,
            name=metric,
            marker_color=colors[index % len(colors)],
            hoverinfo='y'
        ))

    fig.update_layout(
        title='Distribution of Document Ranking Positions',
        xaxis_title='Rank Position of the 1st Retrieved Document per Relevance Label',
        yaxis_title='Number of Queries',
        xaxis=dict(tickmode='array', tickvals=x_indices, ticktext=x_labels),
        barmode='group',
        legend=dict(
            orientation='h',  # horizontal legend
            yanchor='bottom',  # anchor legend to the bottom
            y=1.02,            # position the legend just below the plot
            xanchor='right',   # anchor legend to the right
            x=1                # position legend to the right of the plot
        )
    )
    # Display the plot in Streamlit
    st.plotly_chart(fig)


@st.cache_resource
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
    thresholds = sorted([float(key.split('@')[1]) for key in prec_recall_graphs.keys()])
    precisions = [prec_recall_graphs[f'IPrec(rel={relevance_thres})@{threshold}'] for threshold in thresholds]

    # Add a point for recall=0 to start the plot from the y-axis
    if 0.0 not in thresholds:
        thresholds.insert(0, 0.0)
        precisions.insert(0, prec_recall_graphs.get(f'IPrec(rel={relevance_thres})@0', 0))

    # Create the plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=precisions,
        mode='lines+markers',
        name='Precision-Recall Curve',
        hoverinfo='y',
        line=dict(shape='linear')
    ))

    fig.update_layout(
        title=f'Precision-Recall Curve (Relevance_threshold={relevance_thres})',
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template='plotly_white',

    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


# Generate colors dynamically
def generate_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    return ['rgb' + str(tuple(int(x * 255) for x in colorsys.hsv_to_rgb(*hsv))) for hsv in HSV_tuples]


@st.cache_data
def plot_performance_measures_per_q(data):
    # Extract measures and runs
    eval_measures = list(data[list(data.keys())[0]].keys())
    runs = list(data.keys())

    # Determine the maximum number of queries
    max_queries = max(len(data[run][measure][measure]) for run in runs for measure in eval_measures)

    # Create a subplot for each measure
    fig = make_subplots(rows=len(eval_measures), cols=1, subplot_titles=eval_measures)

    colors = generate_colors(len(runs))

    # Generate patterns dynamically
    base_patterns = ['', '/', '\\', 'x', '-', '|', '+', '.']
    patterns = [base_patterns[i % len(base_patterns)] for i in range(len(runs))]

    # Check if any query ID is longer than 5 characters
    long_labels = any(len(str(x)) > 5 for x in range(1, max_queries + 1))

    for i, measure in enumerate(eval_measures, start=1):
        for j, run in enumerate(runs):
            y_values = data[run][measure][measure]
            x_values = list(range(1, len(y_values) + 1))  # Start from 1 instead of 0

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name=run,
                    marker_color=colors[j],
                    marker_pattern_shape=patterns[j],
                    opacity=0.8,
                    showlegend=(i == 1),  # Only show legend for the first subplot
                    hovertemplate='Query: %{x}<br>Value: %{y:.3f}<extra></extra>'  # Show query number and y-value with 3 decimal places
                ),
                row=i, col=1
            )

        # Update y-axis title and x-axis settings
        fig.update_yaxes(title_text=f"{measure} Value", row=i, col=1)

        if long_labels:
            fig.update_xaxes(
                title_text='Query ID',
                tickmode='array',
                tickvals=list(range(1, max_queries + 1, 5)),  # Show every 5th tick
                ticktext=[str(x) for x in range(1, max_queries + 1, 5)],
                tickangle=90,  # Rotate labels 90 degrees
                range=[0.5, max_queries + 0.5],  # Ensure all bars are visible
                row=i, col=1
            )
        else:
            fig.update_xaxes(
                title_text='Query ID',
                tickmode='linear',
                tick0=1,
                dtick=1,
                range=[0.5, max_queries + 0.5],  # Ensure all bars are visible
                row=i, col=1
            )

    # Update layout
    fig.update_layout(
        height=400 * len(eval_measures),
        title_text="Performance Measures Across Queries",
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
