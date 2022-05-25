import pandas as pd
from tqdm.notebook import tqdm
from cuml.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor


def find_matches(preds, df, n_neighbors=100):
    matcher = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")

    matcher.fit(preds)

    dists, indices = matcher.kneighbors(preds)

    matches = {
        df.index[i]: [df.index[match] for match in indices[i] if match != i]
        for i in range(len(indices))
    }

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
        knn.fit(df_c[cols], df_c.index)
        distances, indices = knn.kneighbors(df_c[cols], return_distance=True)

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


def get_nearest_neighbors(df, n_neighbors=10, cols=["latitude", "longitude"], max_dist=None):

    matcher = NearestNeighbors(n_neighbors=n_neighbors, metric="l1")
    matcher.fit(df[cols])

    distances, indices = matcher.kneighbors(df[cols])

    if max_dist is not None:
        indices = [
            [i for i, d in zip(ids, dists) if d < max_dist]
            for ids, dists in zip(indices, distances)
        ]

    matches = {
        df.index[i]: [df.index[match] for match in indices[i] if match != i]
        for i in tqdm(range(len(indices)))
    }

    return matches


def create_pairs(matches, gt_matches=None):
    pairs = []
    for p1 in matches:
        for p2 in matches[p1]:
            if gt_matches is not None:
                pairs.append([p1, p2, p2 in gt_matches[p1]])
            else:
                pairs.append([p1, p2, False])

    df_pairs = pd.DataFrame(pairs)
    df_pairs.columns = ["id_1", "id_2", "match"]
    return df_pairs
