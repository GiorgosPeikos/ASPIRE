from collections import defaultdict
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
import random
import pandas as pd
from utils.eval_query_collection import get_query_rel_judgements
from utils.eval_query_text_based import query_clf_relevance
from scipy import stats
from collections import Counter
from wordcloud import WordCloud
import json
from typing import Dict, Any
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Function that displays the distribution of ranking position of all retrieved documents based on their relevance label.
@st.cache_resource
def plot_dist_of_retrieved_docs(relevance_ret_pos: dict) -> None:
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
            y=1.02,  # position the legend just below the plot
            xanchor='right',  # anchor legend to the right
            x=1  # position legend to the right of the plot
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


@st.cache_data
# Generate colors dynamically
def generate_colors(n):
    # Predefined list of LaTeX-friendly colors
    predefined_colors = [
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Bluish Green
        '#0072B2',  # Blue
        '#D55E00',  # Vermilion
        '#CC79A7',  # Reddish Purple
        '#000000',  # Black
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Bluish Green
        '#0072B2',  # Blue
        '#CC79A7',  # Reddish Purple
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
    # Extract measures and runs
    eval_measures = list(data[list(data.keys())[0]].keys())
    runs = list(data.keys())

    # Determine the maximum number of queries
    max_queries = max(len(data[run][measure][measure]) for run in runs for measure in eval_measures)

    # Create a subplot for each measure
    fig = make_subplots(rows=len(eval_measures), cols=1, subplot_titles=eval_measures)

    colors = generate_colors(len(runs))

    # Generate patterns dynamically
    base_patterns = ['', '/', 'x', '-', '|', '+', '.']
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
                    hovertemplate='Query: %{x}<br>Difference: %{y:.3f}<br>Run: ' + run + '<extra></extra>'  # Show query number and y-value with 3 decimal places
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
        height=300 * len(eval_measures),
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


@st.cache_data
def plot_performance_difference(data):
    # Extract measures and runs
    eval_measures = list(data[list(data.keys())[0]].keys())
    runs = list(data.keys())

    # Find the baseline run
    baseline_run = next(run for run in runs if "(Baseline)" in run)
    other_runs = [run for run in runs if run != baseline_run]

    # Determine the maximum number of queries
    max_queries = max(len(data[run][measure][measure]) for run in runs for measure in eval_measures)

    # Generate colors and patterns
    colors = generate_colors(len(other_runs))
    base_patterns = ['', '/', 'x', '-', '|', '+', '.']
    patterns = [base_patterns[i % len(base_patterns)] for i in range(len(other_runs))]

    # Check if any query ID is longer than 5 characters
    long_labels = any(len(str(x)) > 5 for x in range(1, max_queries + 1))

    # Create a subplot for each measure
    fig = make_subplots(rows=len(eval_measures), cols=1, subplot_titles=eval_measures, vertical_spacing=0.1)

    for i, measure in enumerate(eval_measures, start=1):
        for j, run in enumerate(other_runs):
            baseline_values = data[baseline_run][measure][measure]
            run_values = data[run][measure][measure]

            # Calculate the difference
            diff_values = [run_val - baseline_val for run_val, baseline_val in zip(run_values, baseline_values)]
            x_values = list(range(1, len(diff_values) + 1))

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=diff_values,
                    name=run,
                    marker_color=colors[j],
                    marker_pattern_shape=patterns[j],
                    opacity=0.8,
                    showlegend=(i == 1),  # Only show legend for the first subplot
                    hovertemplate='Query: %{x}<br>Difference: %{y:.3f}<br>Run: ' + run + '<extra></extra>'
                ),
                row=i, col=1
            )

        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=0.5,
            x1=max_queries + 0.5,
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
            row=i,
            col=1
        )

        # Update y-axis title
        fig.update_yaxes(title_text=f"{measure} Difference", row=i, col=1)

        # Update x-axis settings
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
        height=550 * len(eval_measures),  # Increase height
        title={
            'text': f"""Performance Difference Compared to the Selected Baseline: <span style="color:red;">{baseline_run.replace('(Baseline)', '')}</span>""",
            'x': 0.01,  # Move title to the left
            'xanchor': 'left',
            'yanchor': 'top'
        },
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_performance_and_median_per_experiment(data):
    runs = list(data.keys())
    eval_measures = [measure for measure in data[runs[0]].keys() if not measure.startswith("median_")]

    for run in runs:
        # Create a subplot for each measure within this run
        fig = make_subplots(rows=len(eval_measures), cols=1, subplot_titles=eval_measures)

        max_queries = max(len(data[run][measure][measure]) for measure in eval_measures)

        for i, measure in enumerate(eval_measures, start=1):
            y_values = data[run][measure][measure]  # Access the nested level
            median_values = data[run][f"median_{measure}"]
            x_values = list(range(1, len(y_values) + 1))

            # Add bar plot for performance
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name="Performance",
                    marker_color='blue',
                    opacity=0.8,
                    showlegend=(i == 1),  # Only show legend for the first subplot
                    hovertemplate='Query: %{x}<br>Performance: %{y:.3f}<extra></extra>'
                ),
                row=i, col=1
            )

            # Add scatter plot for median scores
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=median_values,
                    mode='markers',
                    marker=dict(color='red', symbol='star', size=8),
                    name="Median",
                    hovertemplate='Query: %{x}<br>Median: %{y:.3f}<extra></extra>'
                ),
                row=i, col=1
            )

            # Update y-axis title
            fig.update_yaxes(title_text=f"{measure} Value", row=i, col=1)

            # Check if any query ID is longer than 5 characters
            long_labels = any(len(str(x)) > 5 for x in x_values)

            # Update x-axis settings
            if long_labels:
                fig.update_xaxes(
                    title_text='Query ID',
                    tickmode='array',
                    tickvals=list(range(1, max_queries + 1)),  # Show all ticks
                    ticktext=[str(x) for x in range(1, max_queries + 1)],
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
            height=300 * len(eval_measures),
            title_text=f"""Performance of <span style="color:red;">{run}</span> and Comparison to median scores""",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display the plot in Streamlit
        st.write(f"""<center><h4>Analysis of the <span style="color:red;">{run}</span> Experiment</h4></center>""", unsafe_allow_html=True)
        st.write(f"""<center>The median performance per measure, for each query, is computed based on the remaining selected experiments.</center>""", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_performance_difference_threshold(data, threshold):
    # Extract measures and runs
    eval_measures = list(data[list(data.keys())[0]].keys())
    runs = list(data.keys())

    for run in runs:
        # Determine the maximum number of queries for this run
        max_queries = max(len(data[run][measure][measure]) for measure in eval_measures)

        # Create a subplot for each measure within this run
        fig = make_subplots(rows=len(eval_measures), cols=1, subplot_titles=eval_measures, vertical_spacing=0.1)

        for i, measure in enumerate(eval_measures, start=1):
            run_values = data[run][measure][measure]

            # Calculate the difference from the threshold
            diff_values = [run_val - threshold for run_val in run_values]
            x_values = list(range(1, len(diff_values) + 1))

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=diff_values,
                    name=measure,
                    marker_color='blue',
                    opacity=0.8,
                    showlegend=(i == 1),  # Only show legend for the first subplot
                    hovertemplate='Query: %{x}<br>Actual Performance: %{customdata:.3f}<br>Difference: %{y:.3f}<extra></extra>',
                    customdata=run_values  # Add actual performance values for hover
                ),
                row=i, col=1
            )

            # Add a horizontal line at y=0
            fig.add_shape(
                type="line",
                x0=0.5,
                x1=max_queries + 0.5,
                y0=0,
                y1=0,
                line=dict(color="black", width=1, dash="dash"),
                row=i,
                col=1
            )

            # Update y-axis title
            fig.update_yaxes(title_text=f"{measure} Difference", row=i, col=1)

            # Check if any query ID is longer than 5 characters
            long_labels = any(len(str(x)) > 5 for x in x_values)

            # Update x-axis settings
            if long_labels:
                fig.update_xaxes(
                    title_text='Query ID',
                    tickmode='array',
                    tickvals=list(range(1, max_queries + 1)),  # Show all ticks
                    ticktext=[str(x) for x in range(1, max_queries + 1)],
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
            height=550 * len(eval_measures),
            title={
                'text': f"""Performance Difference for <span style="color:red;">{run}</span> w.r.t. the selected threshold: ({threshold:.2f})""",
                'x': 0.01,
                'xanchor': 'left',
                'yanchor': 'top'
            },
            showlegend=False
        )

        # Display the plot in Streamlit
        st.write(f"""<center><h4>Analysis of the <span style="color:red;">{run}</span> Experiment</h4></center>""", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_query_relevance_judgements(selected_qrel):

    relevance_counts, results = get_query_rel_judgements(selected_qrel)

    # Create subplots
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Relevance Judgements per Query'])

    # Create color scale for relevance labels
    relevant_columns = [col for col in relevance_counts.columns if col != 'Irrelevant']
    num_relevant_labels = len(relevant_columns)

    if num_relevant_labels > 1:
        blue_scale = mcolors.LinearSegmentedColormap.from_list("", ["lightblue", "darkblue"])
        blue_colors = [mcolors.rgb2hex(blue_scale(i / (num_relevant_labels - 1))) for i in range(num_relevant_labels)]
    elif num_relevant_labels == 1:
        blue_colors = ["blue"]
    else:
        blue_colors = []

    # Create traces for each relevance level
    for i, column in enumerate(relevance_counts.columns):
        if column == 'Irrelevant':
            color = 'red'
        else:
            color = blue_colors[relevant_columns.index(column)]

        fig.add_trace(
            go.Bar(
                name=column,
                x=[i + 1 for i in range(len(relevance_counts.index))],  # Adjust x-axis to start from 1
                y=relevance_counts[column],
                text=relevance_counts[column],
                textposition='auto',
                marker_color=color,
                hovertemplate='Query: %{x}<br>' + f'{column}: ' + '%{y}<extra></extra>'
            ),
            row=1, col=1
        )

    # Update layout
    fig.update_layout(
        height=550,
        title={
            'text': "Query Relevance Judgements Analysis",
            'x': 0.01,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        barmode='stack',
        xaxis_title="Query ID",
        yaxis_title="Number of Documents",
        legend_title="Relevance Labels",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update x-axis settings
    max_queries = len(relevance_counts.index)
    long_labels = any(len(str(x)) > 5 for x in relevance_counts.index)

    if long_labels:
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, max_queries + 1)),  # Start from 1
            ticktext=[str(x) for x in range(1, max_queries + 1)],  # Start from 1
            tickangle=90,
            range=[0.5, max_queries + 0.5]  # Adjust range
        )
    else:
        fig.update_xaxes(
            tickmode='linear',
            tick0=1,  # Start from 1
            dtick=1,
            range=[0.5, max_queries + 0.5]  # Adjust range
        )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    return results


@st.cache_data
def plot_query_performance_vs_query_length_moving_avg(df, measure):
    # Sort DataFrame by Token Length
    df = df.sort_values('Token Length')

    # Calculate moving average
    window = 20  # Adjust this value to change the smoothness of the line
    df['MA'] = df['Performance'].rolling(window=window, center=True).mean()

    # Create figure
    fig = go.Figure()

    # Add scatter plot for individual queries
    fig.add_trace(go.Scatter(
        x=df['Token Length'],
        y=df['Performance'],
        mode='markers',
        name='Individual Queries',
        marker=dict(
            size=6,
            color=df['Performance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=measure)
        ),
        text=[f"Query ID: {qid}<br>Token Length: {tl}<br>{measure}: {perf:.4f}"
              for qid, tl, perf in zip(df['Query ID'], df['Token Length'], df['Performance'])],
        hoverinfo='text'
    ))

    # Add moving average line
    fig.add_trace(go.Scatter(
        x=df['Token Length'],
        y=df['MA'],
        mode='lines',
        name=f'Moving Average (window={window})',
        line=dict(color='red', width=2)
    ))

    # Update layout
    fig.update_layout(
        xaxis_title="Token Length",
        title={
            'text': f"""<span style="color:red;">{measure}</span> vs Query Length""",
            'x': 0.01,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        yaxis_title=measure,
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


@st.cache_data
def plot_query_performance_vs_query_length_buckets(df, measure):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Equal-width Buckets", "Equal-frequency Buckets"])

    # Function to create hover text
    def create_hover_text(row):
        return (f"Bucket Range: {row.name}<br>"
                f"Mean {measure}: {row['Performance']['mean']:.4f}<br>"
                f"Number of Queries: {row['Performance']['count']}<br>"
                f"Queries: {', '.join(map(str, row['Query ID']))}<br>"
                f"Token Lengths: {', '.join(map(str, row['Token Length']))}")

    # Equal-width buckets
    df['Token Length Bucket'] = pd.cut(df['Token Length'], bins=10)
    grouped_equal_width = df.groupby('Token Length Bucket').agg({
        'Performance': ['mean', 'count'],
        'Query ID': list,
        'Token Length': list
    })

    x_labels_equal_width = [f"{int(interval.left)}-{int(interval.right)}" for interval in grouped_equal_width.index]
    hover_text_equal_width = grouped_equal_width.apply(create_hover_text, axis=1)

    fig.add_trace(
        go.Bar(
            x=x_labels_equal_width,
            y=grouped_equal_width['Performance']['mean'],
            name="Equal-width",
            hovertext=hover_text_equal_width,
            hoverinfo='text',
            text=grouped_equal_width['Performance']['count'],
            textposition='auto'
        ),
        row=1, col=1
    )

    # Equal-frequency buckets
    df['Token Length Bucket'] = pd.qcut(df['Token Length'], q=10, duplicates='drop')
    grouped_equal_freq = df.groupby('Token Length Bucket').agg({
        'Performance': ['mean', 'count'],
        'Query ID': list,
        'Token Length': list
    })

    x_labels_equal_freq = [f"{int(interval.left)}-{int(interval.right)}" for interval in grouped_equal_freq.index]
    hover_text_equal_freq = grouped_equal_freq.apply(create_hover_text, axis=1)

    fig.add_trace(
        go.Bar(
            x=x_labels_equal_freq,
            y=grouped_equal_freq['Performance']['mean'],
            name="Equal-frequency",
            hovertext=hover_text_equal_freq,
            hoverinfo='text',
            text=grouped_equal_freq['Performance']['count'],
            textposition='auto'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Token Length Ranges", tickangle=45)
    fig.update_yaxes(title_text=f"Mean {measure}")

    fig.update_layout(
        height=450,
        showlegend=False,
        title={
            'text': f"""Mean <span style="color:red;">{measure}</span> vs Query Length Buckets""",
            'x': 0.01,
            'xanchor': 'left',
            'yanchor': 'top'
        },
    )
    return fig


@st.cache_data
def plot_query_performance_vs_query_length(data):
    experiments = list(data.keys())
    measures_eval = [measure for measure in data[experiments[0]].keys() if measure != 'token_length']

    for experiment in experiments:
        st.write(f"""<center><h4>Analysis of the <span style="color:red;">{experiment}</span> Experiment</h4></center>""", unsafe_allow_html=True)

        for measure in measures_eval:
            # Create DataFrame with token lengths, performance, and query IDs
            df = pd.DataFrame({
                'Token Length': data[experiment]['token_length']['token_length'],
                'Performance': data[experiment][measure][measure],
                'Query ID': range(1, len(data[experiment][measure][measure]) + 1)
            })

            # Remove rows where Token Length is -1 (placeholder for missing data)
            df = df[df['Token Length'] != -1]

            # Create and display the combined plot
            fig_combined = plot_query_performance_vs_query_length_moving_avg(df, measure)
            st.plotly_chart(fig_combined, use_container_width=True)

            # Bucket analysis
            fig_buckets = plot_query_performance_vs_query_length_buckets(df, measure)
            st.plotly_chart(fig_buckets, use_container_width=True)

            # ANOVA analysis
            f_statistic_ew, p_value_ew = stats.f_oneway(*[group['Performance'].values for name, group in df.groupby(pd.cut(df['Token Length'], bins=10), observed=False)])
            f_statistic_ef, p_value_ef = stats.f_oneway(*[group['Performance'].values for name, group in df.groupby(pd.qcut(df['Token Length'], q=10, duplicates='drop'), observed=False)])

            if p_value_ew < 0.05 or p_value_ef < 0.05:
                st.write(f"""At least one of the bucketing methods shows statistically significant differences between buckets (p < <span style="color: red;">0.05</span>). 
                            ANOVA results (Equal-width buckets): F-statistic = <span style="color: red;">{f_statistic_ew:.4f}</span>, p-value = <span style="color: red;">{p_value_ew:.4f}.</span>
                            ANOVA results (Equal-frequency buckets): F-statistic = <span style="color: red;">{f_statistic_ef:.4f}</span>, p-value = <span style="color: red;">{p_value_ef:.4f}.</span>
                            """, unsafe_allow_html=True)
            else:
                st.write(f"""Neither bucketing method shows statistically significant differences between buckets (p >= <span style="color: red;">0.05</span>).
                            ANOVA results (Equal-width buckets): F-statistic = <span style="color: red;">{f_statistic_ew:.4f}</span>, p-value = <span style="color: red;">{p_value_ew:.4f}.</span>
                            ANOVA results (Equal-frequency buckets): F-statistic = <span style="color: red;">{f_statistic_ef:.4f}</span>, p-value = <span style="color: red;">
                            {p_value_ef:.4f}.</span>""", unsafe_allow_html=True)


@st.cache_data
def create_relevance_wordclouds(query_relevance, queries, method, threshold):

    q_ids_classified = query_clf_relevance(query_relevance, method, threshold)

    st.write(q_ids_classified)

    # write this function so that is prints for each relevance threshold a textual presentation wordcloud if a class has only a query, print the query, if a class has more than 10, sample 10.


    return q_ids_classified
