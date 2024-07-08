import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
import plotly.graph_objects as go


# Function to create a bar plot for evaluation results
def create_evaluation_plot(res_eval, measure_name):
    # Extract query IDs and corresponding values from the Metric objects
    data = [{"Query ID": metric.query_id, "Score": metric.value} for metric in res_eval]

    # Create the plot using Plotly Express
    fig = px.bar(data, x="Query ID", y="Score")
    fig.update_layout(
        xaxis={"categoryorder": "total descending", "tickangle": -45},
        yaxis_title=f"{measure_name}",  # Add your desired y-axis label here
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1.0,
        },
    )

    # Display the figure in Streamlit, embedded in a 2-column layout
    st.plotly_chart(fig)


# Function to plot query scores and add annotations
def plot_queries(scores, ids, median, average, graph_title):
    # Create a DataFrame
    data = {"Scores": scores, "Query ID": ids}
    df = pd.DataFrame(data)

    # Create a scatter plot
    fig = px.scatter(
        df,
        x="Query ID",
        y="Scores",
        color=df["Scores"] > median,
        labels={"color": "Above Median"},
    )

    # Add a vertical line for the median
    fig.add_hline(y=median, line_dash="dash", line_color="green", name="Median")

    # Add a vertical line for the mean
    fig.add_hline(y=average, line_dash="dot", line_color="black", name="Mean")

    # Calculate the total number of "Above Median" and "Below Median" queries
    above_median_count = len([score for score in scores if score > median])
    below_median_count = len([score for score in scores if score <= median])

    # Calculate the total number of "Above Mean" and "Below Mean" queries
    above_mean_count = len([score for score in scores if score > average])
    below_mean_count = len([score for score in scores if score <= average])

    # Create text annotations for the counts
    above_median_annotation = {
        "x": max(ids) + 8,  # Adjust the horizontal position to the left
        "y": max(scores) + 0.1,  # Adjust the vertical position to the top
        "text": f"Above Median: {above_median_count} <br> Above Mean: {above_mean_count}",
        "showarrow": False,
        "xanchor": "left",
        "xshift": 10,  # Adjust the horizontal position
        "bordercolor": "black",
        "borderwidth": 1,
        "borderpad": 4,
        "font": {"size": 12, "color": "black"},
    }

    below_median_annotation = {
        "x": max(ids) + 30,  # Adjust the horizontal position to the left
        "y": min(scores) - 0.2,  # Adjust the vertical position to the top
        "text": f"Below/Eq. Median: {below_median_count} <br> Below/Eq. Mean: {below_mean_count}",
        "showarrow": False,
        "xanchor": "right",
        "xshift": -10,  # Adjust the horizontal position
        "bordercolor": "black",
        "borderwidth": 1,
        "borderpad": 4,
        "font": {"size": 12, "color": "black"},
    }

    # Add the text annotations to the plot
    fig.add_annotation(above_median_annotation)
    fig.add_annotation(below_median_annotation)

    # Update the layout
    fig.update_layout(
        title="Query Performance vs. Median vs. Mean",
        xaxis_title="Query ID",
        yaxis_title=f"{graph_title}",
        legend_title_text="",  # Change this line to rename the legend title
        showlegend=False,
        legend=dict(orientation="h", yanchor="top", xanchor="center"),
    )

    # Customize the scatter points
    fig.update_traces(marker=dict(size=10), text=df["Query ID"], textfont=dict(size=14))

    # Display the figure in Streamlit, embedded in a 2-column layout
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)


def plot_pca(pca_df, classify_queries):
    # Ensure that 'query_id' is a string
    pca_df["query_id"] = pca_df["query_id"].astype(str)

    if classify_queries == "Yes":
        # Create the scatter plot with color based on query difficulty
        fig = px.scatter(
            pca_df,
            x="PCA1",
            y="PCA2",
            color="query_type",  # Color based on query difficulty
            color_discrete_map={
                "Hard Query": " soft red",
                "Easy Query": "blue",
            },  # Define color mapping
            hover_data=["query_id"],  # Include query_id in hover data
            title="2D PCA of Query Embeddings",
            width=800,
            height=600,
        )

        # Custom hover template with larger font for the query ID
        fig.update_traces(
            marker=dict(size=7), hovertemplate="<b>Query ID: %{customdata[0]}</b>"
        )  # Display query ID in bold

        # Update plot layout
        fig.update_layout(
            xaxis_title="PCA1", yaxis_title="PCA2", hoverlabel=dict(font_size=16)
        )  # Increase hover label font size

    st.plotly_chart(fig)


# Function that displays the distribution of ranking position of all retrieved documents based on their relevance label.

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
        '200+': (201, float('inf'))  # Handle values greater than 200
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