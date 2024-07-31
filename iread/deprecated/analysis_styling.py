import pandas as pd


def color_max_min_column(x):
    top_colour = "background-color: #b7f5ae"
    low_colour = "background-color: #f78c81"
    top_score = x.eq(x.max())
    low_score = x.eq(x.min())

    df1 = pd.DataFrame("", index=x.index, columns=x.columns)
    return df1.mask(top_score, top_colour).mask(low_score, low_colour)


def color_max_min_row(x):
    top_colour = "background-color: #b7f5ae"
    low_colour = "background-color: #f78c81"
    top_score = x.T.eq(x.max(axis=1)).T
    low_score = x.T.eq(x.min(axis=1)).T

    df1 = pd.DataFrame("", index=x.index, columns=x.columns)
    return df1.mask(top_score, top_colour).mask(low_score, low_colour)