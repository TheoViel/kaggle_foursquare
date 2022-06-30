import gc
import re
import sys
import difflib
import resource
import Levenshtein
import numpy as np
import pandas as pd
from numba import jit
from collections import defaultdict

from matching import distance, lcs, lcs2, pi, cc_lcs, ll_lcs, pi2
from ressources import dist_by_cat2, dist_by_category_simpl, F_SKIP0, WORDS_C, WORDS_N, CAP_DI


def FE1(p1, p2, size_ratio=1):
    print("- Distances")
    dist = distance(
        np.array(p1["latitude"]),
        np.array(p1["longitude"]),
        np.array(p2["latitude"]),
        np.array(p2["longitude"]),
    )
    df = pd.DataFrame(dist)
    df.columns = ["dist"]
    df["dist"] = df["dist"].astype("int32")
    df["dist1"] = (111173.444444444 * np.abs(p1["latitude"] - p2["latitude"])).astype(
        "int32"
    )  # now on the same scale as dist
    df["dist2"] = np.sqrt(
        np.maximum(0, (1.0 * df["dist"]) ** 2 - df["dist1"] ** 2)
    ).astype(
        "int32"
    )  # get this by subtraction
    for col in ["dist", "dist1", "dist2"]:
        df[col] = np.exp(np.round(np.log(1 + df[col]), 1) - 0.5).astype("int32")
        df[col] = np.minimum(100000, np.round(df[col], -1))
    del dist
    gc.collect()

    # country - categorical
    df["country"] = p1["country"]
    df["country"].loc[p1["country"] != p2["country"]] = 0
    df["country"] = df["country"].astype("category")

    # cat2 - categorical
    df["cat2a"] = np.minimum(p1["cat2"], p2["cat2"]).astype("category")
    df["cat2b"] = np.maximum(p1["cat2"], p2["cat2"]).astype("category")

    ii = np.zeros(df.shape[0], dtype=np.int16)  # integer placeholder
    num_digits2 = 2  # digits for ratios
    for col in ["name", "categories", "address"]:
        print("- Features for column :", col)
        x1, x2 = p1[col].to_numpy(), p2[col].to_numpy()

        # pi1 = partial intersection, start of string
        for i in range(df.shape[0]):
            ii[i] = pi(x1[i], x2[i])
        df[col + "_pi1"] = ii

        # lcs2 = longest common substring
        for i in range(df.shape[0]):
            ii[i] = lcs2(x1[i], x2[i])
        df[col + "_lcs2"] = ii

        # lcs = longest common subsequence
        for i in range(df.shape[0]):
            ii[i] = lcs(x1[i], x2[i])
        df[col + "_lcs"] = ii

        # ll1 - min length of this column
        ll1 = np.maximum(1, np.minimum(p1[col].apply(len), p2[col].apply(len))).astype(
            np.int8
        )  # min length
        ll2 = np.maximum(1, np.maximum(p1[col].apply(len), p2[col].apply(len))).astype(
            np.int8
        )  # max length

        # compound features (ratios) ****************
        # pi1 / ll1 = r1
        df[col + "_pi1_r1"] = np.round((df[col + "_pi1"] / ll1), num_digits2).astype(
            "float32"
        )

        # lcs2 / ll1 = r1
        df[col + "_lcs2_r1"] = np.round((df[col + "_lcs2"] / ll1), num_digits2).astype(
            "float32"
        )

        # lcs2 / ll2 = r2
        df[col + "_lcs2_r2"] = np.round((df[col + "_lcs2"] / ll2), num_digits2).astype(
            "float32"
        )

        # lcs / ll1 = r1
        df[col + "_lcs_r1"] = np.round((df[col + "_lcs"] / ll1), num_digits2).astype(
            "float32"
        )

        # lcs / ll2 = r2
        df[col + "_lcs_r2"] = np.round((df[col + "_lcs"] / ll2), num_digits2).astype(
            "float32"
        )

        # ll1 / ll2 = r3
        df[col + "_r3"] = np.round(ll1 / ll2, num_digits2).astype("float32")

        # lcs2 / lcs = r4
        df[col + "_lcs_r4"] = np.round(
            (df[col + "_lcs2"] / np.maximum(1, df[col + "_lcs"])), num_digits2
        ).astype("float32")
    del x1, x2, ii, ll1, ll2
    gc.collect()

    # NA count for some text columns
    print("- Nan features")
    for col in ["city", "address"]:
        df[col + "_NA"] = (
            (p1[col] == "nan") * 1
            + (p2[col] == "nan") * 1
            + (p1[col] == "") * 1
            + (p2[col] == "") * 1
        ).astype("int8")

    # match for some text columns
    print("- Matching")
    df["phone_m10"] = (
        (p1["phone"] == p2["phone"]) & (p1["phone"] != "") & (p1["phone"] != "nan")
    ).astype("int8")
    df["url_m5"] = (
        (p1["url"].str[:5] == p2["url"].str[:5])
        & (p1["url"] != "")
        & (p1["url"] != "nan")
    ).astype("int8")

    # simp cat match
    print("- Category match")
    mask1 = (p1["category_simpl"] >= 1) & (p1["category_simpl"] == p2["category_simpl"])
    mask2 = (p1["category_simpl"] == 0) & (p1["categories"] == p2["categories"])
    df["same_cat_simpl"] = ((mask1) | (mask2)).astype("int8")
    del mask1, mask2
    gc.collect()

    print("- Ratios")
    # ratio of dist to mean dist by cat2
    df["dm"] = df["cat2a"].astype("int32").map(dist_by_cat2)
    df["dist_r1"] = np.minimum(
        800, np.exp(np.round(np.log(1 + df["dist"] / df["dm"]), 1)) - 1
    ).astype(
        "float32"
    )  # median: log scale, cap at 800

    # ratio of dist to mean dist by category_simpl
    df["cat_simpl"] = p1["category_simpl"].astype("int16")
    df["cat_simpl"].iloc[df["same_cat_simpl"] == 0] = 0  # not a match - make it 0
    df["cat_simpl"] = df["cat_simpl"].astype("category")
    df["dm"] = df["cat_simpl"].astype("int32").map(dist_by_category_simpl)
    df["dist_r2"] = np.minimum(
        800, np.exp(np.round(np.log(1 + df["dist"] / df["dm"]), 1)) - 1
    ).astype(
        "float32"
    )  # median: log scale, cap at 800
    df.drop(["dm", "cat_simpl"], axis=1, inplace=True)
    p1 = p1[["id", "name"]]
    p2 = p2[["id", "name"]]
    gc.collect()

    # number of times col appears in this data
    print("- Count encodings")
    cc_cap = 10000  # cap on counts
    for col in ["id", "name"]:
        p12 = p1[[col]].append(
            p2[[col]], ignore_index=True
        )  # count it in both p1 and p2
        df1 = p12[col].value_counts()
        p1["cc"] = p1[col].map(df1) * size_ratio
        p2["cc"] = p2[col].map(df1) * size_ratio
        del p12, df1
        gc.collect()
        # features
        df[col + "_cc_min"] = np.minimum(
            cc_cap, np.minimum(p1["cc"], p2["cc"])
        )  # min, capped at X, scaled
        df[col + "_cc_max"] = np.minimum(
            cc_cap, np.maximum(p1["cc"], p2["cc"])
        )  # max, capped at X, scaled
        # log-transform
        df[col + "_cc_min"] = (
            np.exp(np.round(np.log(df[col + "_cc_min"] + 1), 1)) - 0.5
        ).astype("int16")
        df[col + "_cc_max"] = (
            np.exp(np.round(np.log(df[col + "_cc_max"] + 1), 1)) - 0.5
        ).astype("int16")
        p1.drop("cc", axis=1, inplace=True)
        p2.drop("cc", axis=1, inplace=True)
    return df


def dfs(res, node, connected_components):
    # global connected_components
    if node not in connected_components:
        connected_components[node] = set()
        for next_ in res[node]:
            dfs(res, next_, connected_components)
            connected_components[node] = connected_components[next_]
        connected_components[node].add(node)


def get_strongly_connected_components(graph):
    seen = set()
    components = []
    for node in graph:
        if node not in seen:
            component = []
            nodes = {node}
            while nodes:
                node = nodes.pop()
                seen.add(node)
                component.append(node)
                nodes.update(graph[node].difference(seen))
            components.append(component)
    return components


@jit
def count_close(lat, lon, lat1, lon1):
    cc = np.zeros([lat1.shape[0], 7], dtype=np.int32)
    d0 = 2000**2 / 111111**2
    d1 = 1000**2 / 111111**2
    d2 = 500**2 / 111111**2
    d3 = 200**2 / 111111**2
    d4 = 100**2 / 111111**2
    d5 = 50**2 / 111111**2
    d6 = 5000**2 / 111111**2
    for i in range(lat1.shape[0]):
        m = np.cos(lat1[i]) ** 2
        dist2 = (lat - lat1[i]) ** 2 + m * (lon - lon1[i]) ** 2
        dist2 = dist2[dist2 < d6]  # select subset of data to save time
        cc[i, 0] = (dist2 < d0).sum()
        cc[i, 1] = (dist2 < d1).sum()
        cc[i, 2] = (dist2 < d2).sum()
        cc[i, 3] = (dist2 < d3).sum()
        cc[i, 4] = (dist2 < d4).sum()
        cc[i, 5] = (dist2 < d5).sum()
        cc[i, 6] = (dist2 < d6).sum()
    return cc


@jit
def count_close_cat(lat, lon, lat1, lon1, cat, cat1, cat2):
    cc = np.zeros([lat1.shape[0], 7], dtype=np.int32)
    d0 = 2000**2 / 111111**2
    d1 = 1000**2 / 111111**2
    d2 = 500**2 / 111111**2
    d3 = 200**2 / 111111**2
    d4 = 100**2 / 111111**2
    d5 = 50**2 / 111111**2
    d6 = 5000**2 / 111111**2
    for i in range(lat1.shape[0]):
        m = np.cos(lat1[i]) ** 2
        dist2 = (lat - lat1[i]) ** 2 + m * (lon - lon1[i]) ** 2
        dist2 = dist2[
            (cat == cat1[i]) | (cat == cat2[i])
        ]  # only keep points for which cat matches
        dist2 = dist2[dist2 < d6]  # select subset of data to save time
        cc[i, 0] = (dist2 < d0).sum()
        cc[i, 1] = (dist2 < d1).sum()
        cc[i, 2] = (dist2 < d2).sum()
        cc[i, 3] = (dist2 < d3).sum()
        cc[i, 4] = (dist2 < d4).sum()
        cc[i, 5] = (dist2 < d5).sum()
        cc[i, 6] = (dist2 < d6).sum()
    return cc


def ratio_of_similar_elements(L1, L2):
    common = [x for x in L1 if x in L2]
    a = 2*len(common) - len(L1) - len(L2)
    nb = len(L1) + len(L2) - len(common)
    if nb > 0:
        return a, a / nb
    else:
        return 0, 0


def find_num_in_string(s):
    L = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
    L = [''.join([x for x in s if x.isdigit()]) for s in L]
    return L


def FE2(df, p1, p2, train, ressources_path="", size_ratio=1):
    print('- Cat links & quantiles')
    cat_links = pd.read_pickle(
        ressources_path + "link_between_categories.pkl"
    )  # link-between-categories
    Dist_quantiles = pd.read_pickle(
        ressources_path + "Dist_quantiles_per_catpairs.pkl"
    )  # dist-quantiles-per-cat

    cat_links_ratio, cat_links_ratio_all = [], []
    dist_qtl = []

    p1 = p1[["id"]].merge(train[["id", "categories_split"]], on="id", how="left")
    p2 = p2[["id"]].merge(train[["id", "categories_split"]], on="id", how="left")
    for i, (L1, L2) in enumerate(zip(p1["categories_split"], p2["categories_split"])):
        # Find the biggest score corresponding to one of the possible category-pairs
        s0, s1 = 0, 0
        q = [Dist_quantiles[("", "")].copy()]  # default : couple of nan
        for cat1 in L1:
            for cat2 in L2:
                key = tuple(sorted([cat1, cat2]))
                if key in cat_links:
                    x = cat_links[key]
                    if x[0] > s0:
                        s0 = x[0]
                    if x[1] > s1:
                        s1 = x[1]
                if key in Dist_quantiles:
                    q.append(Dist_quantiles[key])
        # Append
        cat_links_ratio.append(s0)
        cat_links_ratio_all.append(s1)
        dist_qtl.append(np.max(np.array(q), axis=0))

    # Drop useless column
    # train.drop(columns = ["categories_split"], inplace=True)
    del Dist_quantiles, cat_links
    gc.collect()

    # add other columns - needed for FE
    p1 = p1[["id"]].merge(train, on="id", how="left")
    p2 = p2[["id"]].merge(train, on="id", how="left")

    # Raise recursion limit

    resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
    sys.setrecursionlimit(10**8)

    # 1. Simply connected components
    print('- Simply connected components')

    # Create graph
    graph = defaultdict(set)
    for id1, id2 in zip(p1["id"], p2["id"]):
        if id1 != id2:
            graph[id1].add(id2)
            graph[id2].add(id1)

    # Find connected comoponents
    connected_components = defaultdict(set)
    for node_ in graph:
        dfs(graph, node_, connected_components)
    connected_components = map(tuple, connected_components.values())
    unique_components = set(connected_components)
    Connexes = [list(x) for x in unique_components]

    Len_connect = {}
    for Liste_index in Connexes:
        for idx in Liste_index:
            Len_connect[idx] = min(len(Liste_index), 200)

    # Add feature
    df["Nb_connect1"] = p1["id"].apply(lambda idx: Len_connect[idx]).astype("int16")
    df["Nb_connect2"] = p2["id"].apply(lambda idx: Len_connect[idx]).astype("int16")

    # Ratio with Nb_multiPoi
    eps, MAX = 1e-1, 1e5
    df["ratio_connect_multipoi1"] = df["Nb_connect1"] / (df["Nb_multiPoi_1"] + eps)
    df["ratio_connect_multipoi2"] = df["Nb_connect2"] / (df["Nb_multiPoi_2"] + eps)

    df["ratio_connect_multipoi1"] = df["ratio_connect_multipoi1"].apply(
        lambda x: x if x <= MAX else MAX
    ).astype('float32')
    df["ratio_connect_multipoi2"] = df["ratio_connect_multipoi2"].apply(
        lambda x: x if x <= MAX else MAX
    ).astype('float32')

    del Connexes, connected_components  # , Len_connect
    gc.collect()

    # 2. Strongly connected components
    print('- Strongly connected components')

    # Strongly connected components
    Connexes = get_strongly_connected_components(graph)

    Len_strong_connect = {}
    for Liste_index in Connexes:
        for idx in Liste_index:
            Len_strong_connect[idx] = min(len(Liste_index), 200)

    # Add feature
    df["Nb_strong_connect"] = (
        p1["id"].apply(lambda x: Len_strong_connect[x]).astype("int16")
    )

    # Ratio with nb_multiPoi
    eps, MAX = 1e-1, 1e5
    df["ratio_strong_connect_multipoi1"] = df["Nb_strong_connect"] / (df["Nb_multiPoi_1"] + eps)
    df["ratio_strong_connect_multipoi2"] = df["Nb_strong_connect"] / (df["Nb_multiPoi_2"] + eps)
    # Avoid too high values
    df["ratio_strong_connect_multipoi1"] = df["ratio_strong_connect_multipoi1"].apply(
        lambda x: x if x <= MAX else MAX
    ).astype('float32')
    df["ratio_strong_connect_multipoi2"] = df["ratio_strong_connect_multipoi2"].apply(
        lambda x: x if x <= MAX else MAX
    ).astype('float32')

    del Connexes, graph  # , Len_strong_connect
    gc.collect()

    # Cat link scores
    print('- Cat link score')

    df["cat_link_score"] = cat_links_ratio.copy()
    df["cat_link_score_all"] = cat_links_ratio_all.copy()

    col_cat_distscores = ["Nb_multiPoi", "mean", "q25", "q50", "q75", "q90", "q99"]

    for x in ["cat_link_score", "cat_link_score_all"]:
        df[x] = np.round(df[x], 3).astype("float32")
    df.loc[:, [x + "_pair" for x in col_cat_distscores]] = dist_qtl.copy()
    for x in col_cat_distscores:
        df[x + "_pair"] = df[x + "_pair"].astype("float32")
    del cat_links_ratio, cat_links_ratio_all, dist_qtl

    # Add features
    for x in col_cat_distscores + [
        "cat_solo_score",
        "freq_pairing_with_other_groupedcat",
    ]:
        df[x + "_1"] = np.maximum(p1[x], p2[x]).astype("float32")
        df[x + "_2"] = np.minimum(p1[x], p2[x]).astype("float32")

    # Feature engineering : dist divided by previous features
    MAX, eps = 1e6, 1e-6
    for x in col_cat_distscores[1:]:
        df[x + "_ratiodist_1"] = df["dist"] / (
            df[x + "_1"] + eps
        )  # epsilon to avoid dividing by 0
        df[x + "_ratiodist_2"] = df["dist"] / (
            df[x + "_2"] + eps
        )  # epsilon to avoid dividing by 0
        df[x + "_ratiodist_pair"] = df["dist"] / (
            df[x + "_pair"] + eps
        )  # epsilon to avoid dividing by 0
        # Avoid too high values
        df[x + "_ratiodist_1"] = (
            df[x + "_ratiodist_1"]
            .apply(lambda x: x if x <= MAX else MAX)
            .astype("float32")
        )
        df[x + "_ratiodist_2"] = (
            df[x + "_ratiodist_2"]
            .apply(lambda x: x if x <= MAX else MAX)
            .astype("float32")
        )
        df[x + "_ratiodist_pair"] = (
            df[x + "_ratiodist_pair"]
            .apply(lambda x: x if x <= MAX else MAX)
            .astype("float32")
        )
    # Useless after all (low fi; it's better to only keep dist ratio)
    df.drop(columns=[x + "_pair" for x in col_cat_distscores], inplace=True)

    # Link between grouped categories
    print("- Link between grouped categories")
    cat_links = pd.read_pickle(ressources_path + "link_between_grouped_categories.pkl")
    L1, L2 = [], []
    for cat1, cat2 in zip(p1["category_simpl"], p2["category_simpl"]):
        key = tuple(sorted([cat1, cat2]))
        x = [0, 0]
        if key in cat_links:
            x = cat_links[key]
        L1.append(x[0])
        L2.append(x[1])
    df["grouped_cat_link_score"] = L1.copy()
    df["grouped_cat_link_score_all"] = L2.copy()
    for x in ["grouped_cat_link_score", "grouped_cat_link_score_all"]:
        df[x] = np.round(df[x], 3).astype("float32")

    del L1, L2, cat_links
    gc.collect()

    ################################
    drops = [
        "freq_pairing_with_other_groupedcat",
        "cat_solo_score",
        "Nb_multiPoi",
        "mean",
        "q25",
        "q50",
        "q75",
        "q90",
        "q99",
    ]
    p1.drop(drops, axis=1, inplace=True)
    p2.drop(drops, axis=1, inplace=True)

    ii = np.zeros(df.shape[0], dtype=np.int16)  # integer placeholder
    fi = np.zeros(df.shape[0], dtype=np.float32)  # floating point placeholder
    fi2 = np.zeros(df.shape[0], dtype=np.float32)  # floating point placeholder2
    num_digits = 3  # digits for floats
    num_digits2 = 2  # digits for ratios

    # list of features to skip due to low fi (<0.00015)
    # f_skip1 = set(df.columns)
    to_skip = set(df.columns) & set(F_SKIP0)

    for col in [
        # "name_initial",
        # "name_initial_decode",
        # "nameC",
        "name",
        # "categories",
        # "address",
        # "url",
        # "city",
        # "state",
        # "zip",
        # "phone",
    ]:
        print("- Features for column", col)
        x1, x2 = p1[col].to_numpy(), p2[col].to_numpy()
        dfp = pd.concat([p1[[col]], p2[col]], 1)
        dfp.columns = [col + "_1", col + "_2"]

        name = col + "_cclcs"  # cclcs = count of common substrings of length X+
        if name not in to_skip:  # skip to save time
            # for i in range(df.shape[0]):
            #     ii[i] = cc_lcs(x1[i], x2[i], 4)
            # df[name] = ii
            df[name] = dfp[[col + "_1", col + "_2"]].parallel_apply(
                lambda x: cc_lcs(x[0], x[1], 4), axis=1
            )

        name = col + "_lllcs"  # lllcs = total length of common substrings of length X+
        if name not in to_skip:  # skip to save time
            min_len = 1 if col == "nameC" else 5

            # for i in range(df.shape[0]):
            #     ii[i] = ll_lcs(x1[i], x2[i], min_len)
            # df[name] = ii

            df[name] = dfp[[col + "_1", col + "_2"]].parallel_apply(
                lambda x: ll_lcs(x[0], x[1], min_len), axis=1
            )

        name = col + "_lcs2"  # lcs2 = longest common substring
        if name not in to_skip:  # skip to save time
            # for i in range(df.shape[0]):
            #     ii[i] = lcs2(x1[i], x2[i])
            # df[name] = ii

            df[name] = dfp[[col + "_1", col + "_2"]].parallel_apply(
                lambda x: lcs2(x[0], x[1]), axis=1
            )

        name = col + "_lcs"  # lcs = longest common subsequence
        if name not in to_skip:  # skip to save time
            # for i in range(df.shape[0]):
            #     ii[i] = lcs(x1[i], x2[i])
            # df[name] = ii

            df[name] = dfp[[col + "_1", col + "_2"]].parallel_apply(
                lambda x: lcs(x[0], x[1]), axis=1
            )

        name = col + "_pi1"  # pi1 = partial intersection, start of string
        if name not in to_skip:  # skip to save time
            for i in range(df.shape[0]):
                ii[i] = pi(x1[i], x2[i])
            df[name] = ii
            # df[name] = dfp[[col + "_1", col + "_2"]].parallel_apply(
            #     lambda x: pi(x[0], x[1]), axis=1
            # )

        name = col + "_pi2"  # pi2 = partial intersection, end of string
        if name not in to_skip:  # skip to save time
            # for i in range(df.shape[0]):
            #     ii[i] = pi2(x1[i], x2[i])
            # df[name] = ii

            df[name] = dfp[[col + "_1", col + "_2"]].parallel_apply(
                lambda x: pi2(x[0], x[1]), axis=1
            )

        name = col + "_ld"  # ld = Levenshtein.distance
        if name not in to_skip:  # skip to save time
            for i in range(df.shape[0]):
                ii[i] = Levenshtein.distance(x1[i], x2[i])
            df[name] = ii

        name = col + "_ljw"  # ljw = Levenshtein.jaro_winkler (float)
        if name not in to_skip:  # skip to save time
            for i in range(df.shape[0]):
                fi[i] = Levenshtein.jaro_winkler(x1[i], x2[i])

            df[name] = np.round(fi, num_digits).astype(np.float32)  # round

        # dsm = difflib.SequenceMatcher (float); not symmetrical, do apply twice!
        # for i in range(df.shape[0]):
        #     fi[i] = difflib.SequenceMatcher(None, x1[i], x2[i]).ratio()
        # for i in range(df.shape[0]):
        #     fi2[i] = difflib.SequenceMatcher(None, x2[i], x1[i]).ratio()

        fi = dfp[[col + "_1", col + "_2"]].parallel_apply(
            lambda x: difflib.SequenceMatcher(None, x[0], x[1]).ratio(), axis=1
        )

        if col + "_dsm1" not in to_skip:
            df[col + "_dsm1"] = np.round(fi, num_digits).astype(np.float32)

        # ll1 - min length of this column
        ll1 = np.maximum(1, np.minimum(p1[col].apply(len), p2[col].apply(len))).astype(
            np.int8
        )  # min length
        ll2 = np.maximum(1, np.maximum(p1[col].apply(len), p2[col].apply(len))).astype(
            np.int8
        )  # max length
        df[col + "_ll1"] = ll1

        # compound features (ratios) ****************
        # pi1 / ll1 = r1
        if col + "_pi1_r1" not in to_skip:
            df[col + "_pi1_r1"] = np.round((df[col + "_pi1"] / ll1), num_digits2).astype(
                "float32"
            )

        # pi2 / ll1 = r1
        if col + "_pi2_r1" not in to_skip:
            df[col + "_pi2_r1"] = np.round((df[col + "_pi2"] / ll1), num_digits2).astype(
                "float32"
            )

        # lcs2 / ll1 = r1
        if col + "_lcs2_r1" not in to_skip:
            df[col + "_lcs2_r1"] = np.round((df[col + "_lcs2"] / ll1), num_digits2).astype(
                "float32"
            )

        # lcs2 / ll2 = r2
        if col + "_lcs2_r2" not in to_skip:
            df[col + "_lcs2_r2"] = np.round((df[col + "_lcs2"] / ll2), num_digits2).astype(
                "float32"
            )

        # lcs / ll1 = r1
        if col + "_lcs_r1" not in to_skip:
            df[col + "_lcs_r1"] = np.round((df[col + "_lcs"] / ll1), num_digits2).astype(
                "float32"
            )

        # lcs / ll2 = r2
        if col + "_lcs_r2" not in to_skip:
            df[col + "_lcs_r2"] = np.round((df[col + "_lcs"] / ll2), num_digits2).astype(
                "float32"
            )

        # lllcs / ll1 = r1
        if col + "_lllcs_r1" not in to_skip:
            df[col + "_lllcs_r1"] = np.round(
                (df[col + "_lllcs"] / ll1), num_digits2
            ).astype("float32")

        # lllcs / ll2 = r2
        if col + "_lllcs_r2" not in to_skip:
            df[col + "_lllcs_r2"] = np.round(
                (df[col + "_lllcs"] / ll2), num_digits2
            ).astype("float32")

        # ll1 / ll2 = r3
        if col + "_r3" not in to_skip:
            df[col + "_r3"] = np.round(ll1 / ll2, num_digits2).astype("float32")

        # lcs2 / lcs = r4
        if col + "_lcs_r4" not in to_skip:
            df[col + "_lcs_r4"] = np.round(
                (df[col + "_lcs2"] / np.maximum(1, df[col + "_lcs"])), num_digits2
            ).astype("float32")

        # count of NAs
        df[col + "_NA"] = (
            (p1[col] == "nan") * 1
            + (p2[col] == "nan") * 1
            + (p1[col] == "") * 1
            + (p2[col] == "") * 1
        ).astype("int8")

        # match5 - if not NA
        df[col + "_m5"] = (
            (p1[col].str[:5] == p2[col].str[:5])
            & (p1[col] != "")
            & (p2[col] != "")
            & (p1[col] != "nan")
            & (p2[col] != "nan")
        ).astype("int8")

        # match10 - if not NA
        df[col + "_m10"] = (
            (p1[col].str[:10] == p2[col].str[:10])
            & (p1[col] != "")
            & (p2[col] != "")
            & (p1[col] != "nan")
            & (p2[col] != "nan")
        ).astype("int8")

        # break

    # drop skipped columns
    df.drop(list(F_SKIP0.intersection(df.columns)), axis=1, inplace=True)
    del x1, x2, fi, fi2, ll1, ll2
    gc.collect()

    # cap features - after ratios. To reduce overfitting (cap determined to impact <0.1% of cases)
    for col in CAP_DI.keys():
        if col in df.columns:
            df[col] = np.minimum(df[col], CAP_DI[col])

    # do something to reduce cardinality of ratios
    # round some floats to 2 digits, not 3
    for col in [
        "name_initial_ljw",
        "name_initial_decode_ljw",
        "name_initial_dsm1",
        "name_initial_dsm2",
        "name_initial_decode_dsm1",
        "name_initial_decode_dsm2",
        "zip_ljw",
        "address_dsm1",
        "address_dsm2",
        "address_lcs_r4",
        "address_ljw",
        "categories_dsm1",
        "categories_dsm2",
        "categories_ljw",
        "city_dsm1",
        "city_dsm2",
        "city_ljw",
        "name_dsm1",
        "name_dsm2",
        "name_ljw",
        "nameC_dsm1",
        "nameC_dsm2",
        "nameC_ljw",
        "state_dsm1",
        "state_dsm2",
        "state_ljw",
        "url_dsm1",
        "url_dsm2",
        "url_ljw",
    ]:
        if col in df.columns:
            df[col] = np.round(df[col], 2)

    # log-transform/round some features to reduce cardinality
    for col in [
        "q75_ratiodist_pair",
        "q90_ratiodist_pair",
        "mean_ratiodist_pair",
        "q50_ratiodist_pair",
        "q25_ratiodist_pair",
        "q99_ratiodist_pair",
        "q50_ratiodist_2",
        "q90_ratiodist_2",
        "q25_ratiodist_2",
        "q25_ratiodist_1",
        "q75_ratiodist_2",
        "mean_ratiodist_2",
        "q99_ratiodist_2",
        "q75_ratiodist_1",
        "q50_ratiodist_1",
        "q90_ratiodist_1",
        "q99_ratiodist_1",
        "mean_ratiodist_1",
    ]:
        if col in df.columns:
            df[col] = np.exp(np.round(np.log(df[col] + 1), 1))

    # number of times col appears in this data
    print('- Count encoding')
    cc_cap = 10000  # cap on counts
    for col in [
        "name", "address", "categories", "id", "city", "state",
        "zip", "phone", "city_group", "state_group"
    ]:
        p12 = p1[[col]].append(
            p2[[col]], ignore_index=True
        )  # count it in both p1 and p2
        df1 = p12[col].value_counts()
        p1["cc"] = p1[col].map(df1) * size_ratio
        p2["cc"] = p2[col].map(df1) * size_ratio
        del p12, df1
        gc.collect()
        # features
        df[col + "_cc_min"] = np.minimum(
            cc_cap, np.minimum(p1["cc"], p2["cc"])
        )  # min, capped at X, scaled
        df[col + "_cc_max"] = np.minimum(
            cc_cap, np.maximum(p1["cc"], p2["cc"])
        )  # max, capped at X, scaled
        # log-transform
        df[col + "_cc_min"] = (
            np.exp(np.round(np.log(df[col + "_cc_min"] + 1), 1)) - 0.5
        ).astype("int16")
        df[col + "_cc_max"] = (
            np.exp(np.round(np.log(df[col + "_cc_max"] + 1), 1)) - 0.5
        ).astype("int16")
        p1.drop("cc", axis=1, inplace=True)
        p2.drop("cc", axis=1, inplace=True)

    # drop unneeded features to same memory
    p1.drop(["url", "city", "state", "zip", "phone", "country"], axis=1, inplace=True)
    p2.drop(["url", "city", "state", "zip", "phone", "country"], axis=1, inplace=True)

    # find words in categories
    print('- Words in categories')
    w1 = np.zeros([p1.shape[0], len(WORDS_C)], dtype=np.int8)
    w2 = np.zeros([p2.shape[0], len(WORDS_C)], dtype=np.int8)
    for i, word in enumerate(WORDS_C):  # currently 90
        w1[p1["categories"].str.contains(word, regex=False), i] = 1
        w2[p2["categories"].str.contains(word, regex=False), i] = 1
        # features: match for each word. Only for words with high fi.
        # Word order matters - do not change it.
        if i in [22]:  # college
            df["word_c_" + str(i) + "_m"] = (w1[:, i] * w2[:, i]).astype("int8")

    df["word_c_m_cc"] = (w1 * w2).sum(axis=1).astype("int8")  # count of matches
    df["word_c_cs"] = np.nan_to_num(
        np.round(
            (w1 * w2).sum(axis=1)
            / np.sqrt((w1 * w1).sum(axis=1) * (w2 * w2).sum(axis=1)),
            num_digits,
        )
    ).astype(
        "float32"
    )  # cosine similarity

    # find words in name
    print('- Words in names')
    w1 = np.zeros([p1.shape[0], len(WORDS_N)], dtype=np.int8)
    w2 = np.zeros([p2.shape[0], len(WORDS_N)], dtype=np.int8)
    for i, word in enumerate(WORDS_N):  # currently 51
        w1[p1["name"].str.contains(word, regex=False), i] = 1
        w2[p2["name"].str.contains(word, regex=False), i] = 1

    df["word_n_m_cc"] = (w1 * w2).sum(axis=1).astype("int8")  # count of matches
    df["word_n_cs"] = np.nan_to_num(
        np.round(
            (w1 * w2).sum(axis=1)
            / np.sqrt((w1 * w1).sum(axis=1) * (w2 * w2).sum(axis=1)),
            num_digits,
        )
    ).astype(
        "float32"
    )  # cosine similarity
    del w1, w2, ii
    gc.collect()

    # feature: count of ids within X meters of current. 140 seconds.
    print('- Close count')
    lat = np.array(train["latitude"])
    lon = np.array(train["longitude"])
    lat1 = np.array((p1["latitude"] + p2["latitude"]) / 2)
    lon1 = np.array((p1["longitude"] + p2["longitude"]) / 2)

    cc = np.zeros([lat1.shape[0], 7], dtype=np.int32)
    step = 1
    for long_min in range(-180, 190, step):
        idx1 = (lon1 > long_min) & (lon1 < long_min + step)
        idx2 = (lon > long_min - 0.1) & (lon < long_min + step + 0.1)  # margin
        if idx1.sum() > 0 and idx2.sum() > 0:
            cc1 = count_close(lat[idx2], lon[idx2], lat1[idx1], lon1[idx1])
            cc[idx1, :] = cc1

    cc = cc * size_ratio
    cc = np.minimum(cc, 10000).astype(np.int16)  # scale
    cc = (np.exp(np.round(np.log(cc + 1), 1)) - 0.5).astype(
        np.int16
    )  # reduce cardinality - log-transform and round - 56 unique vals to 1000
    df["id_cc_2K"] = cc[:, 0]
    df["id_cc_1K"] = cc[:, 1]
    df["id_cc_500"] = cc[:, 2]
    df["id_cc_200"] = cc[:, 3]
    df["id_cc_100"] = cc[:, 4]
    df["id_cc_50"] = cc[:, 5]
    df["id_cc_5K"] = cc[:, 6]

    # feature: count of ids of the same cat2 within X meters of current. 250 seconds.
    print('- Close count of same category2')
    cat = np.array(train["cat2"])
    cat1 = np.array(p1["cat2"])
    cat2 = np.array(p2["cat2"])

    for long_min in range(-180, 190, step):
        idx1 = (lon1 > long_min) & (lon1 < long_min + step)
        idx2 = (lon > long_min - 0.1) & (lon < long_min + step + 0.1)  # margin
        if idx1.sum() > 0 and idx2.sum() > 0:
            cc1 = count_close_cat(
                lat[idx2],
                lon[idx2],
                lat1[idx1],
                lon1[idx1],
                cat[idx2],
                cat1[idx1],
                cat2[idx1],
            )
            cc[idx1, :] = cc1
    cc = cc * size_ratio
    cc = np.minimum(cc, 10000).astype(np.int16)  # scale
    cc = (np.exp(np.round(np.log(cc + 1), 1)) - 0.5).astype(
        np.int16
    )  # reduce cardinality - log-transform and round - 56 unique vals to 1000
    df["id_cc_cat_2K"] = cc[:, 0]
    df["id_cc_cat_1K"] = cc[:, 1]
    df["id_cc_cat_500"] = cc[:, 2]
    df["id_cc_cat_200"] = cc[:, 3]
    df["id_cc_cat_100"] = cc[:, 4]
    df["id_cc_cat_50"] = cc[:, 5]
    df["id_cc_cat_5K"] = cc[:, 6]

    # feature: count of ids of the same 'category_simpl' within X meters of current.
    print('- Close count of same category_simpl')
    cat = np.array(train["category_simpl"])
    cat1 = np.array(p1["category_simpl"])
    cat2 = np.array(p2["category_simpl"])
    # if category_simpl = 0 then use categories
    train["cat_orig"] = (
        train["categories"].astype("category").cat.codes + cat.max() + 10
    )  # make sure this does not intersect with category_simpl codes
    cat[cat == 0] = np.array(train["cat_orig"])[cat == 0]
    p1 = p1.merge(train[["id", "cat_orig"]], on="id", how="left")
    p2 = p2.merge(train[["id", "cat_orig"]], on="id", how="left")
    cat1[cat1 == 0] = np.array(p1["cat_orig"])[cat1 == 0]
    cat2[cat2 == 0] = np.array(p2["cat_orig"])[cat2 == 0]
    train.drop("cat_orig", axis=1, inplace=True)

    for long_min in range(-180, 190, step):
        idx1 = (lon1 > long_min) & (lon1 < long_min + step)
        idx2 = (lon > long_min - 0.1) & (lon < long_min + step + 0.1)  # margin
        if idx1.sum() > 0 and idx2.sum() > 0:
            cc1 = count_close_cat(
                lat[idx2],
                lon[idx2],
                lat1[idx1],
                lon1[idx1],
                cat[idx2],
                cat1[idx1],
                cat2[idx1],
            )
            cc[idx1, :] = cc1
    cc = cc * size_ratio
    cc = np.minimum(cc, 10000).astype(np.int16)  # scale
    cc = (np.exp(np.round(np.log(cc + 1), 1)) - 0.5).astype(
        np.int16
    )  # reduce cardinality - log-transform and round - 56 unique vals to 1000
    df["id_cc_simplcat_2K"] = cc[:, 0]
    df["id_cc_simplcat_1K"] = cc[:, 1]
    df["id_cc_simplcat_500"] = cc[:, 2]
    df["id_cc_simplcat_200"] = cc[:, 3]
    df["id_cc_simplcat_100"] = cc[:, 4]
    df["id_cc_simplcat_50"] = cc[:, 5]
    df["id_cc_simplcat_5K"] = cc[:, 6]

    # feature: compare numeric part of the name/address
    print('- Compare numeric part of the name/address')
    for col in ["name", "address"]:
        n1num = p1[col].apply(lambda x: "".join(re.findall(r"\d+", x)))
        n1num[n1num == ""] = "0"
        n1num = n1num.str[:9].astype("int32")
        n2num = p2[col].apply(lambda x: "".join(re.findall(r"\d+", x)))
        n2num[n2num == ""] = "0"
        n2num = n2num.str[:9].astype("int32")
        df[col + "_num"] = (
            (n1num != 0) * 1 + (n2num != 0) * 1 + ((n2num == n1num) & (n1num != 0)) * 2
        ).astype(
            "category"
        )  # 0/1/2/4
    del n1num, n2num
    gc.collect()

    # language combination of name
    df["langs"] = (
        np.minimum(p1["lang"], p2["lang"]) + np.maximum(p1["lang"], p2["lang"]) * 3
    ).astype(
        "category"
    )  # 6 possible combinations

    # cat_simpl, if matches; as cat.
    df["cat_simpl"] = p1["category_simpl"].astype("int16")
    df["cat_simpl"].iloc[df["same_cat_simpl"] == 0] = 0  # not a match - make it 0
    df["cat_simpl"] = df["cat_simpl"].astype("category")

    # Numbers in names
    print('- Number in names features')
    Num_in_p1_names = [sorted(find_num_in_string(text)) for text in p1['name_initial_decode']]
    Num_in_p2_names = [sorted(find_num_in_string(text)) for text in p2['name_initial_decode']]

    Num_similar = []
    Nb_of_similar_num = []
    Ratio_similar_num = []
    for i, num1 in enumerate(Num_in_p1_names):
        num2 = Num_in_p2_names[i]
        if num1 == [] and num2 == []:
            Num_similar.append(0)
        elif num1 == [] or num2 == []:
            Num_similar.append(-1)
        else:
            if num1 == num2:
                Num_similar.append(3)
            elif all(x in num2 for x in num1) or all(x in num1 for x in num2):
                Num_similar.append(2)
            elif any(x in num2 for x in num1) or any(x in num1 for x in num2):
                Num_similar.append(1)
            else:
                Num_similar.append(-2)

        nb, ratio = ratio_of_similar_elements(num1, num2)
        Nb_of_similar_num.append(nb)
        Ratio_similar_num.append(ratio)

    # num in names
    df['num_in_name'] = Num_similar.copy()
    df['num_in_name'] = df['num_in_name'].astype('int16')
    df['nb_in_name'] = Nb_of_similar_num.copy()
    df['nb_in_name'] = df['nb_in_name'].astype('int16')
    df['ratio_in_name'] = Ratio_similar_num.copy()
    df['ratio_in_name'] = df['ratio_in_name'].astype('float32')

    del Num_similar, Nb_of_similar_num, Ratio_similar_num
    gc.collect()

    # drop skipped columns
    df.drop(list(F_SKIP0.intersection(df.columns)), axis=1, inplace=True)
    df.drop("same_cat_simpl", axis=1, inplace=True)
    gc.collect()
    return df
