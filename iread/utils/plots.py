import plotly.express as px
import streamlit as st
import pandas as pd


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
