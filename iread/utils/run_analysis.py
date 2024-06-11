import pandas as pd


def find_unjudged(run: pd.DataFrame, qrels: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    run["rank"] = run["rank"].astype(int)
    run = run[run["rank"] < cutoff]

    df = pd.merge(run, qrels, how="left", on=["doc_id", "query_id"])
    df.fillna(-1, inplace=True)

    return df[df["relevance"] == -1]
