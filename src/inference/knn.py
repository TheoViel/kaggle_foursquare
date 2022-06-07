import cuml
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors


def find_matches(preds, df, n_neighbors=100):
    matcher = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")

    matcher.fit(preds)

    dists, indices = matcher.kneighbors(preds)

    ids = df.index[indices.flatten()].values.reshape(-1, n_neighbors)
    df["matches"] = list(ids)
    matches = df[["matches"]].to_dict(orient="dict")["matches"]
    for k in matches:
        matches[k] = [m for m in matches[k] if m != k]

    return matches


def get_nearest_neighbors(
    df, n_neighbors=10, cols=["latitude", "longitude"], max_dist=None
):

    matcher = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric="l1")
    matcher.fit(df[cols].values)

    distances, indices = matcher.kneighbors(df[cols].values)

    ids = df.index.to_pandas()[indices.get().flatten()].values.reshape(-1, n_neighbors)
    df["matches"] = list(ids)
    matches = df[["matches"]].to_pandas().to_dict(orient="dict")["matches"]

    if max_dist is not None:
        df["dists"] = list(distances.get())
        dists = df[["dists"]].to_pandas().to_dict(orient="dict")["dists"]
        for k in matches:
            matches[k] = [m for m, d in zip(matches[k], dists[k]) if m != k and d < max_dist]

    else:
        for k in matches:
            matches[k] = [m for m in matches[k] if m != k]

    return matches


def get_nearest_neighbors_swapped(
    df, n_neighbors=10, cols=["latitude", "longitude"], max_dist=None
):

    matcher = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric="l1")
    matcher.fit(df[cols].values)

    distances, indices = matcher.kneighbors(df[cols].values)
    distances_s, indices_s = matcher.kneighbors(df[cols[::-1]].values)

    distances = np.concatenate([distances, distances_s[:, :n_neighbors // 2]], 1)
    indices = np.concatenate([indices, indices_s[:, :n_neighbors // 2]], 1)

    orders = np.argsort(distances, 1)[:, :n_neighbors]
    indices = [ids[order] for ids, order in zip(indices, orders)]

#     if max_dist is not None:
#         indices = [
#             [i for i, d in zip(ids, dists) if d < max_dist]
#             for ids, dists in zip(indices, distances)
#         ]

    ids = df.index.to_pandas()[np.concatenate(indices).get()].values.reshape(-1, n_neighbors)
    df["matches"] = list(ids)
    matches = df[["matches"]].to_pandas().to_dict(orient="dict")["matches"]
    for k in matches:
        matches[k] = [m for m in matches[k] if m != k]

    return matches


def get_nearest_neighbors_sklearn(
    df, n_neighbors=10, cols=["latitude", "longitude"], max_dist=None
):
    matcher = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="minkowski",
        radius=1.0,
        algorithm="kd_tree",
        leaf_size=30,
        p=2,
        n_jobs=-1,
    )
    matcher.fit(df[cols].values)

    distances, indices = matcher.kneighbors(df[cols].values)

    if max_dist is not None:
        indices = [
            [i for i, d in zip(ids, dists) if d < max_dist]
            for ids, dists in zip(indices, distances)
        ]

    ids = df.index[indices.flatten()].values.reshape(-1, n_neighbors)
    df["matches"] = list(ids)
    matches = df[["matches"]].to_dict(orient="dict")["matches"]
    for k in matches:
        matches[k] = [m for m in matches[k] if m != k]

    return matches


def get_nearest_neighbors_country(
    df, n_neighbors=10, cols=["latitude", "longitude"], max_dist=None
):

    matches = {}
    for country, df_c in tqdm(df.groupby("country")):
        n_neighbors_ = min(len(df_c), n_neighbors)

        knn = KNeighborsRegressor(
            n_neighbors=n_neighbors_, metric="haversine", n_jobs=-1
        )
        knn.fit(np.radians(df_c[cols]), df_c.index)
        distances, indices = knn.kneighbors(np.radians(df_c[cols]), return_distance=True)

        if max_dist is not None and len(df_c) > n_neighbors:
            indices = [
                [i for i, d in zip(ids, dists) if d < max_dist]
                for ids, dists in zip(indices, distances)
            ]

        matches_country = {
            df_c.index[i]: [df_c.index[match] for match in indices[i] if match != i]
            for i in range(len(indices))
        }
        matches.update(matches_country)
    return matches


def create_pairs(nn_matches, naive_matches, n_neighbors=20, gt_matches=None):
    pairs = []
    for p1, matches_naive in naive_matches.items():
        matches_nn = nn_matches.get(p1, [])

        for i, p2 in enumerate(matches_naive):
            rank = i if i < n_neighbors else -0.5  # -1
            label = p2 in gt_matches[p1] if gt_matches is not None else False

            try:
                rank_nn = matches_nn.index(p2)
            except ValueError:
                rank_nn = -1

            pairs.append([p1, p2, label, rank, rank_nn])

        for i, p2 in enumerate(matches_nn):
            if p2 in matches_naive:
                continue
            rank = i
            label = p2 in gt_matches[p1] if gt_matches is not None else False

            pairs.append([p1, p2, label, -1, i])

    df_pairs = pd.DataFrame(pairs)
    df_pairs.columns = ["id_1", "id_2", "match", "rank", "rank_nn"]
    return df_pairs


def find_phone_matches(id_, df):
    number = df['phone'][id_]

#     matches = list(df[df['phone'] == number].index)
    matches = list(df[df['phone'].apply(lambda x: x in number or number in x)].index)
    matches.remove(id_)
    return matches


def find_url_matches(id_, df):
    url = df['url'][id_]

#     matches = list(df[df['url'] == url].index)
    matches = list(df[df['url'].apply(lambda x: x in url or url in x)].index)
    matches.remove(id_)
    return matches
