import gc
import numpy as np
import pandas as pd

from fe import FE2, FE1
from fe_theo import feature_engineering_theo, feature_engineering_theo_2
from dtypes import DTYPES_1, DTYPES_2, convert_to_dtypes


def feature_engineering_1(p1, p2, size_ratio=1):
    # Vincent & Youri
    df = FE1(p1, p2)
    df = convert_to_dtypes(df, DTYPES_1)

    df.insert(0, "id_1", p1["id"].values)
    df.insert(1, "id_2", p2["id"].values)

    # Théo
    cols = [
        "id",
        "name",
        "latitude",
        "longitude",
        "address",
        "country",
        "url",
        "phone",
        "city",
        "state",
        "zip",
        "categories",
        "idx",
    ]
    pairs = pd.concat([p1[cols], p2[cols]], axis=1)
    pairs.columns = [c + "_1" for c in cols] + [c + "_2" for c in cols]

    del p1, p2
    gc.collect()

    df_theo, _ = feature_engineering_theo(pairs)
    df_theo = convert_to_dtypes(df_theo, DTYPES_1)

    del pairs
    gc.collect()

    # Merge
    df_merged = df.merge(df_theo, on=["id_1", "id_2"])

    return df_merged


def feature_engineering_2(df_p, train, ressources_path="", size_ratio=1):
    # Vincent & Youri
    df_p = df_p.merge(train[["id", "Nb_multiPoi"]], left_on="id_1", right_on="id").drop(
        "id", axis=1
    )
    df_p = df_p.merge(
        train[["id", "Nb_multiPoi"]], left_on="id_2", right_on="id", suffixes=("_1", "_2")
    ).drop("id", axis=1)

    p1 = df_p[["id_1"]].copy()
    p1.columns = ["id"]
    p2 = df_p[["id_2"]].copy()
    p2.columns = ["id"]

    df = FE2(df_p, p1, p2, train, ressources_path, size_ratio=size_ratio)
    df = convert_to_dtypes(df, DTYPES_2)

    # Théo
    cols = ["id", "name", "address", "city", "state", "zip", "url", "phone", "idx"]

    for col in cols[1:]:
        train.loc[train[col] == "", col] = np.nan

    pairs = pd.concat([
        p1[["id"]].merge(train[cols], on="id", how="left")[cols],
        p2[["id"]].merge(train[cols], on="id", how="left")[cols]
    ], axis=1)
    pairs.columns = [c + "_1" for c in cols] + [c + "_2" for c in cols]

    del p1, p2
    gc.collect()

    df_theo, _ = feature_engineering_theo_2(train, pairs, cuda=False)
    df_theo = convert_to_dtypes(df_theo, DTYPES_2)

    del pairs
    gc.collect()

    # Merge
    df_merged = df.merge(df_theo, on=["id_1", "id_2"])

    cols_to_end = ["point_of_interest_1", "fold_1", "point_of_interest_2", "fold_2", "match"]
    cols_to_end = [c for c in cols_to_end if c in df_merged.columns]

    if len(cols_to_end):
        to_end = df_merged[cols_to_end]
        df_merged.drop(cols_to_end, axis=1, inplace=True)
        df_merged[cols_to_end] = to_end

    return df_merged
