import cudf
import Levenshtein
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz


# String Distances

def levenshtein(a, b):
    return Levenshtein.distance(a, b)


def jaro(a, b):
    return Levenshtein.jaro_winkler(a, b)


def wratio(a, b):
    return fuzz.WRatio(a, b)


def partial_ratio(a, b):
    return fuzz.partial_ratio(a, b)


def compute_string_distance(fct, a, b):
    if a != "" and b != "":
        return fct(a.lower(), b.lower())
    else:
        return np.nan


def compute_string_distances(df, string_columns, verbose=0):
    functions = [
        levenshtein,
    ]

    features = []
    for col in string_columns:
        df[col + "_1"].fillna("", inplace=True)
        df[col + "_2"].fillna("", inplace=True)

        df[f"{col}_len_1"] = df[col + "_1"].apply(len)
        df[f"{col}_len_2"] = df[col + "_2"].apply(len)

        df[f"{col}_len_diff"] = np.abs(df[f"{col}_len_1"] - df[f"{col}_len_2"])
        features.append(f"{col}_len_diff")

        for fct in functions:
            name = fct.__name__
            if verbose:
                print(f"- Column : {col}  -  Function : {name}")

            df[col + "_" + name] = df[[col + "_1", col + "_2"]].apply(
                lambda x: compute_string_distance(fct, x[0], x[1]), axis=1
            )
            features.append(col + "_" + name)

        # Normalize
        df[f"{col}_levenshtein"] = df[f"{col}_levenshtein"] / df[
            [f"{col}_len_1", f"{col}_len_2"]
        ].max(axis=1)

    return features


def compute_string_distances_2(df, string_columns, verbose=0):
    functions = [wratio, partial_ratio]

    features = []
    for col in string_columns:
        df[col + "_1"].fillna("", inplace=True)
        df[col + "_2"].fillna("", inplace=True)

        for fct in functions:
            name = fct.__name__
            if verbose:
                print(f"- Column : {col}  -  Function : {name}")

            df[col + "_" + name] = df[[col + "_1", col + "_2"]].apply(
                lambda x: compute_string_distance(fct, x[0], x[1]), axis=1
            )
            features.append(col + "_" + name)

    return features


# Other string utils

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
        return np.array(sims.get()).flatten()
    except AttributeError:
        return np.array(sims).flatten()


# Coordinates Distances

def angular_distance(lat1, long1, lat2, long2):
    delta_long = pd.concat([np.abs(long2 - long1), 360 - np.abs(long2 - long1)], 1).min(
        1
    )
    return np.abs(lat2 - lat1) + delta_long


def angular_distance_l2(lat1, long1, lat2, long2):
    delta_long = pd.concat([np.abs(long2 - long1), 360 - np.abs(long2 - long1)], 1).min(
        1
    )
    return np.sqrt(np.clip((lat2 - lat1) ** 2 + delta_long**2, 0, 100000))


def compute_position_distances(df, functions):
    features = []

    lats_1 = df["latitude_1"]
    longs_1 = df["longitude_1"]
    lats_2 = df["latitude_2"]
    longs_2 = df["longitude_2"]

    for fct in functions:
        name = fct.__name__
        df[name] = fct(lats_1, longs_1, lats_2, longs_2)
        df[name + "_s"] = fct(longs_1, lats_1, lats_2, longs_2)
        df[name + "_min"] = df[[name, name + "_s"]].min(1)

        df.drop([name, name + "_s"], axis=1, inplace=True)

        features += [name + "_min"]

    return features


# NaNs

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

    return features


# Level 1

def feature_engineering_theo(df_p):
    features = []

    print("- Computing position distances")
    DIST_FCTS = [angular_distance, angular_distance_l2]
    features += compute_position_distances(df_p, DIST_FCTS)

    if not isinstance(df_p, pd.DataFrame):
        df_p = df_p.to_pandas()

    FEATURES_SAME = [
        ("state", is_equal),
        ("zip", is_included),
        ("city", is_included),
    ]

    for col, fct in FEATURES_SAME:
        print(f"- Computing feature same_{col}")
        df_p[f"same_{col}"] = (
            df_p[[f"{col}_1", f"{col}_2"]]
            .fillna("")
            .apply(lambda x: fct(x[0], x[1]), axis=1)
            .astype(float)
        )

        features.append(f"same_{col}")

    STRING_DIST_COLS = ["name", "address", "url"]
    features += compute_string_distances(df_p, STRING_DIST_COLS, verbose=1)

    # to_keep = ["id_1", "id_2"] + features
    # df_p.drop([c for c in df_p.columns if c not in to_keep], axis=1, inplace=True)

    return df_p, features


# Level 2

def feature_engineering_theo_2(df, df_p, cuda=False):
    features = []

    NAN_COLS = [
        "address",
        "city",
        "state",
        "zip",
        "url",
        "phone"
    ]

    STRING_COLS = [
        "name",
        "address",
        "url",
    ]
    TF_IDF_PARAMS = [
        ((3, 3), "char_wb"),  # char trigrams
    ]

    print("- Computing nan features")
    features += compute_nan_features(df_p, NAN_COLS)

    # TF-idf
    for col in STRING_COLS:
        for ngram_range, analyzer in TF_IDF_PARAMS:
            ft_name = f"{col}_tf_idf_{ngram_range[0]}{ngram_range[1]}_{analyzer}_sim"
            print(f"- Computing feature {ft_name}")

            if cuda:
                from cuml.feature_extraction.text import TfidfVectorizer

                tf_idf = TfidfVectorizer(
                    use_idf=False, ngram_range=ngram_range, analyzer=analyzer
                )
                if isinstance(df_p, pd.DataFrame):
                    tf_idf_mat = tf_idf.fit_transform(
                        cudf.from_pandas(df[col].fillna("nan"))
                    )
                else:
                    tf_idf_mat = tf_idf.fit_transform(df[col].fillna("nan"))

            else:
                from sklearn.feature_extraction.text import TfidfVectorizer

                tf_idf = TfidfVectorizer(
                    use_idf=False, ngram_range=ngram_range, analyzer=analyzer
                )
                tf_idf_mat = tf_idf.fit_transform(df[col].fillna("nan"))

            df_p[ft_name] = tf_idf_similarity(df_p, tf_idf_mat)
            df_p.loc[
                df_p[col + "_1"].isna() | df_p[col + "_2"].isna(), ft_name
            ] = np.nan
            features.append(ft_name)

    if not isinstance(df_p, pd.DataFrame):
        df_p = df_p.to_pandas()

    # String distances
    features += compute_string_distances_2(df_p, STRING_COLS, verbose=1)

    to_keep = ["id_1", "id_2"] + features
    df_p.drop([c for c in df_p.columns if c not in to_keep], axis=1, inplace=True)

    return df_p, features
