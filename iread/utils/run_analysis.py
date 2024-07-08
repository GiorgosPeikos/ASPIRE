import pandas as pd
import streamlit as st


@st.cache_data
def return_available_measures():
    default_measures = [
        "AP@100",  # Average Precision
        "P@10",  # Precision
        "nDCG@10",  # Normalized Discounted Cumulative Gain
        "R@50",  # Recall
        "RR@1000",  # Reciprocal Rank
    ]

    overall_measures = [
        "NumQ",  # The total number of queries.
        "NumRet",  # Number of Retrieved Documents
        "NumRel",  # Number of Relevant Documents
        "NumRelRet",  # Number of Relevant Retrieved Documents (Alias)
    ]

    precision_measures = [
        "P@5",  # Number of Relevant Retrieved Documents (Alias)
        "P@10",
        "P@25",
        "P@50",
        "P@100",
        "Rprec"
    ]

    freq_measures = [
        "AP@10",  # Average Precision
        "AP@100",  # Average Precision
        "AP@1000",  # Average Precision
        "nDCG@5",  # Normalized Discounted Cumulative Gain
        "nDCG@10",  # Normalized Discounted Cumulative Gain
        "nDCG@1000",  # Normalized Discounted Cumulative Gain
        "P@5",  # Precision
        "P@10",  # Precision
        "P@25",  # Precision
        "R@10",  # Recall
        "R@25",  # Recall
        "R@50",  # Recall
        "R@1000",  # Recall
        "RR@1000",  # Reciprocal Rank
        "Rprec",  # Precision at R
        "Bpref",  # Binary Preference
        "infAP",  # Inferred Average Precision
        "NumRelRet",  # Number of Relevant Retrieved Documents (Alias)
        "NumRel",
        "NumRet",
    ]

    custom_user = [
        "AP",  # Average Precision
        "nDCG",  # Normalized Discounted Cumulative Gain
        "P",  # Precision
        "R",  # Recall
        "RR",  # Reciprocal Rank
        "Rprec",  # Precision at R
        "Bpref",  # Binary Preference
        "infAP",  # Inferred Average Precision
        "Judged",

    ]

    rest_measures = [
        # "Accuracy",
        # "alpha_DCG",
        # "alpha_nDCG",
        # "AP_IA",
        # "BPM",
        "Compat",
        # "ERR",
        # "ERR_IA",
        # "INSQ",
        # "INST",
        # "IPrec",
        "Judged",
        # "nERR_IA",
        # "nNRBP",
        # "NRBP",
        "NumQ",
        "RBP",
        # "SDCG",
        "SetAP",
        "SetF",
        "SetP",
        "SetR",
        # "StRecall",
        # "Success",
        # "α_DCG",
        # "α_nDCG"
    ]
    return freq_measures, rest_measures, custom_user, default_measures, overall_measures, precision_measures


# The function gets the qrels and the run from the session and a selected metric from the user and returns the
# evaluation measure.
# Define a set of measures that support a relevance threshold based on the documentation
# https://ir-measur.es/en/latest/measures.html

# Define a set of measures that support relevance level e.g. (rel=2)
measures_with_rel_param = {
    "Accuracy",
    "alpha_DCG",
    "alpha_nDCG",
    "AP",
    "AP_IA",
    "Bpref",
    "ERR_IA",
    "infAP",
    "IPrec",
    "NERR8",
    "NERR9",
    "NERR10",
    "NERR11",
    "nERR_IA",
    "nNRBP",
    "NRBP",
    "P",
    "P_IA",
    "R",
    "RBP",
    "Rprec",
    "RR",
    "SetAP",
    "SetF",
    "SetP",
    "SetR",
    "NumRet",
    "NumRelRet",
}

# Define a set of measures that support a cutoff e.g. @10
measures_with_cutoff = {
    "Accuracy",
    "alpha_DCG",
    "alpha_nDCG",
    "AP",
    "AP_IA",
    "BPM",
    "ERR",
    "ERR_IA",
    "nDCG",
    "NERR8",
    "NERR9",
    "NERR10",
    "NERR11",
    "nERR_IA",
    "nNRBP",
    "NRBP",
    "P",
    "P_IA",
    "R",
    "RBP",
    "RR",
    "SDCG",
    "SetAP",
    "SetF",
    "SetP",
    "SetR",
    "StRecall",
    "Success",
}

# Define a set of measures that need to be return as int
measures_int = ["NumRelRet", "NumRel", "NumRet", "NumQ"]


def find_unjudged(run: pd.DataFrame, qrels: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    run["rank"] = run["rank"].astype(int)
    run = run[run["rank"] < cutoff]

    df = pd.merge(run, qrels, how="left", on=["doc_id", "query_id"])
    df.fillna(-1, inplace=True)

    return df[df["relevance"] == -1]

