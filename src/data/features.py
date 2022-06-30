import cudf
import torch
import pylcs
import difflib
import Levenshtein
import numpy as np
from fuzzywuzzy import fuzz


def lcs(a, b):
    return pylcs.lcs(a, b)


def gesh(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def levenshtein(a, b):
    return Levenshtein.distance(a, b)


def jaro(a, b):
    return Levenshtein.jaro_winkler(a, b)


def wratio(a, b):
    return fuzz.WRatio(a, b)


def partial_ratio(a, b):
    return fuzz.partial_ratio(a, b)


def category_similarity(a, b):
    set_a = set(str(a).split(", "))
    set_b = set(str(b).split(", "))
    return len(set_a.intersection(set_b)) / min(len(a), len(b))


def compute_string_distance(fct, a, b):
    if a != "" and b != "":
        return fct(a.lower(), b.lower())
    else:
        return np.nan


def compute_string_distances(df, string_columns, verbose=0):
    functions = [
        # lcs,
        gesh,
        levenshtein,
        jaro,
        wratio,
        partial_ratio
    ]

    features = []
    for col in string_columns:
        df[col + "_1"].fillna("", inplace=True)
        df[col + "_2"].fillna("", inplace=True)

        df[f"{col}_len_1"] = df[col + "_1"].parallel_apply(len)
        df[f"{col}_len_2"] = df[col + "_2"].parallel_apply(len)

        df[f"{col}_len_diff"] = np.abs(df[f"{col}_len_1"] - df[f"{col}_len_2"])
        features.append(f"{col}_len_diff")

        for fct in functions:
            name = fct.__name__
            if verbose:
                print(f"- Column : {col}  -  Function : {name}")

            df[col + "_" + name] = df[[col + "_1", col + "_2"]].parallel_apply(
                lambda x: compute_string_distance(fct, x[0], x[1]), axis=1
            )
            features.append(col + "_" + name)

        # Normalized cols
        df[f"{col}_levenshtein_n"] = df[f"{col}_levenshtein"] / df[
            [f"{col}_len_1", f"{col}_len_2"]
        ].max(axis=1)
        df[f"{col}_lcs_n1"] = df[f"{col}_lcs"] / df[f"{col}_len_1"]
        df[f"{col}_lcs_n2"] = df[f"{col}_lcs"] / df[f"{col}_len_1"]

        features += [f"{col}_levenshtein_n", f"{col}_lcs_n1", f"{col}_lcs_n2"]

    if verbose:
        print('- Categories similarity')
    df["categories_sim"] = df[["categories_1", "categories_2"]].parallel_apply(
        lambda x: compute_string_distance(category_similarity, x[0], x[1]), axis=1
    )
    features.append('categories_sim')

    return features


def haversine_distance(lat1, long1, lat2, long2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(long2 - long1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return 6371.0088 * dist


def manhattan_distance(lat1, long1, lat2, long2):
    return np.abs(lat2 - lat1) + np.abs(long2 - long1)


def euclidian_distance(lat1, long1, lat2, long2):
    return np.sqrt(np.clip((lat2 - lat1) ** 2 + (long2 - long1) ** 2, 0, 100000))


def angular_distance(lat1, long1, lat2, long2):
    delta_long = cudf.concat(
        [np.abs(long2 - long1), 360 - np.abs(long2 - long1)], 1
    ).min(1)
    return np.abs(lat2 - lat1) + delta_long


def angular_distance_l2(lat1, long1, lat2, long2):
    delta_long = cudf.concat(
        [np.abs(long2 - long1), 360 - np.abs(long2 - long1)], 1
    ).min(1)
    return np.sqrt(np.clip((lat2 - lat1) ** 2 + delta_long**2, 0, 100000))


def is_included(a, b):
    if a == "" or b == "":
        return np.nan
    else:
        return a.lower() in b.lower() or b.lower() in a.lower()


def is_equal(a, b):
    if a == "" or b == "":
        return np.nan
    else:
        return a.lower() == b.lower()


def tf_idf_similarity(pairs, matrix):
    i1s = pairs["idx_1"].values.astype(int).tolist()
    i2s = pairs["idx_2"].values.astype(int).tolist()
    sims = matrix[i1s].multiply(matrix[i2s]).sum(1)

    try:
        return sims.get()
    except AttributeError:
        return sims


def compute_nn_distances(df, matrix, suffix=""):
    i1s = df["idx_1"].values.tolist()
    i2s = df["idx_2"].values.tolist()

    df["nn_dist_l1" + suffix] = (
        torch.abs(matrix[i1s] - matrix[i2s])
    ).mean(1).cpu().numpy()
    df["nn_dist_l2" + suffix] = torch.sqrt(
        ((matrix[i1s] - matrix[i2s]) ** 2).mean(1)
    ).cpu().numpy()
    df["nn_cosine_sim" + suffix] = (
        (matrix[i1s] * matrix[i2s]).sum(1) /
        (torch.sqrt((matrix[i1s] ** 2).sum(1)) * torch.sqrt((matrix[i2s] ** 2).sum(1)))
    ).cpu().numpy()

    return ["nn_dist_l1" + suffix, "nn_dist_l2" + suffix, "nn_cosine_sim" + suffix]


def compute_nan_features(df, cols):
    features = []
    to_fill = []
    for col in cols:
        for suffix in ["_1", "_2"]:
            df[f"{col + suffix}_nan"] = df[col + suffix].isna().astype(np.uint8)
            to_fill.append(col + suffix)
        df[f"{col}_both_nan"] = df[[f"{col}_1_nan", f"{col}_2_nan"]].min(axis=1)
        df[f"{col}_any_nan"] = df[[f"{col}_1_nan", f"{col}_2_nan"]].max(axis=1)

        features += [f"{col}_any_nan", f"{col}_both_nan"]

    df["info_power_1"] = df[[f"{col}_1_nan" for col in cols]].lt(1).mean(axis=1)
    df["info_power_2"] = df[[f"{col}_2_nan" for col in cols]].lt(1).mean(axis=1)
    df["info_diff"] = np.abs(df["info_power_1"] - df["info_power_2"])

    for col in cols:
        df.drop([f"{col}_1_nan", f"{col}_2_nan"], axis=1, inplace=True)
    features += ["info_power_1", "info_power_2", "info_diff"]

    df[to_fill].fillna("", inplace=True)
    return features


def compute_position_distances(df):
    features = []

    lats_1 = df["latitude_1"]
    longs_1 = df["longitude_1"]
    lats_2 = df["latitude_2"]
    longs_2 = df["longitude_2"]

    # df["longitude_diff"] = np.abs(longs_2 - longs_1)
    # df["latitude_diff"] = np.abs(lats_2 - lats_1)
    # df["long2_lat1"] = np.abs(longs_2 - lats_1)
    # df["long1_lat2"] = np.abs(lats_2 - longs_1)
    # features = +["longitude_diff", "latitude_diff", "long2_lat1", "long1_lat2"]

    functions = [
        haversine_distance,
        manhattan_distance,
        angular_distance,
        euclidian_distance,
        angular_distance_l2,
    ]

    for fct in functions:
        name = fct.__name__
        df[name] = fct(lats_1, longs_1, lats_2, longs_2)
        df[name + "_s"] = fct(longs_1, lats_1, lats_2, longs_2)
        df[name + "_min"] = df[[name, name + "_s"]].min(1)

        features += [name, name + "_s", name + "_min"]

    return features


def nli_features(df):
    features = []

    # Params
    FEATURES_SAME = [
        ("phone", is_included),
        ("url", is_included),
    ]

    STRING_DIST_COLS = ["url"]

    # Distances
    features += compute_position_distances(df)
    df = df.to_pandas()

    # Inclusion / equality features
    for col, fct in FEATURES_SAME:
        df[f"same_{col}"] = (
            df[[f"{col}_1", f"{col}_2"]]
            .parallel_apply(lambda x: fct(x[0], x[1]), axis=1)
            .astype(float)
        )
        df[f"same_{col}"].fillna(-1, inplace=True)
        features.append(f"same_{col}")

    # String features
    features += compute_string_distances(df, STRING_DIST_COLS, verbose=1)

    # Normalize
    to_normalize = [f for f in features if "same" not in f]
    stats = {}
    for col in to_normalize:
        mean_ = df[col].mean()
        std_ = df[col].std()
        stats[col] = (mean_, std_)

        df[col] = (df[col] - mean_) / (std_)

    return df, features, stats
