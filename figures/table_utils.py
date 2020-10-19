import pandas as pd
import numpy as np
import os
import re
import copy


def round_float_cols(df):
    new_df = copy.copy(df)
    for col in new_df.columns:
        if new_df[col].dtype == float and (
            "mean" in col or "std" in col or "threshold" not in col
        ):
            if new_df[col].max() <= 1:
                new_df[col] = new_df[col] * 100
                new_df[col] = new_df[col].round(1)
            else:
                new_df[col] = new_df[col].round(2)
    return new_df


def mean_std_notation(df):
    new_df = copy.copy(df)
    for col in new_df.columns:
        if col.endswith("mean"):
            std_col = col[0:-4] + "std"
            new_df[col] = new_df[col].astype(str)
            new_df[col] = new_df[col] + " (" + new_df[std_col].astype(str) + ")"
            new_df.drop(std_col, axis=1, inplace=True)
    return new_df


def get_latex_col(df, col, sub=None, sup=None):
    new_df = copy.copy(df)
    if sub and sup:
        new_df[col] = (
            "$"
            + new_df[col]
            + "_{"
            + new_df[sub].astype(str)
            + "}^{"
            + new_df[sup].astype(str)
            + "}$"
        )
        new_df.drop([sub, sup], axis=1, inplace=True)
    elif sub and not sup:
        new_df[col] = "$" + new_df[col] + "_{" + new_df[sub].astype(str) + "}$"
        new_df.drop(sub, axis=1, inplace=True)
    elif sup and not sub:
        new_df[col] = "$" + new_df[col] + "^{" + new_df[sub].astype(str) + "}$"
        new_df.drop([sup], axis=1, inplace=True)
    return new_df


def generate_pk_col(df, col1, col2, col3=None):
    new_df = copy.copy(df)
    if col3:
        newcol = (
            new_df[col1].astype(str)
            + new_df[col2].astype(str)
            + new_df[col3].astype(str)
        )
    else:
        newcol = new_df[col1].astype(str) + new_df[col2].astype(str)
    new_df["pk"] = newcol
    return new_df


def compute_mean_std(data, cols):
    df = copy.copy(data)
    df = (
        df.groupby(["model_name", "threshold_pw", "iou_threshold"])
        .agg([np.mean, np.std])[cols]
        .reset_index()
    )
    df.columns = df.columns.droplevel(0)
    print(df.columns)
    df = df.rename_axis(None, axis=1)
    new_cols = ["model_name", "threshold_pw", "iou_threshold"]
    old = df.columns.tolist()
    for c in old:
        col = cols[0]
        if c == "mean":
            new_cols.append(col + "_mean")
        elif c == "std":
            new_cols.append(col + "_std")
            cols.pop(0)

        continue
    df.columns = new_cols
    return df


def get_arch_col(data):
    df = copy.copy(data)
    fcn = df.loc[df.model_name.str.endswith("s"), :]
    sw = df.loc[df.model_name.str.startswith("s"), :]
    fcn["architecture"] = "FCN"
    sw["architecture"] = "SW"
    return pd.concat([fcn, sw])