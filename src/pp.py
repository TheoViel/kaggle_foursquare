import numpy as np
import pandas as pd


def match_pois(df2, threshold=0.5, threshold_small=0.5, threshold_big=0.5):
    matches = df2[df2["match"] > threshold].reset_index(drop=True)

    id1, id2, preds = np.split(matches.values, [1, 2], axis=1)
    id1, id2, preds = id1.flatten(), id2.flatten(), preds.flatten()

    id_to_poi = {}  # maps each ID to POI
    poi_counts = (
        {}
    )  # counts number of IDs in each POI - used for threshold determination
    poi_to_id = {}

    poi = 0
    for i, (i1, i2, pred) in enumerate(zip(id1, id2, preds)):
        if (i1 in id_to_poi) and (i2 in id_to_poi):
            # Merging will be handled later
            continue

        # i1 is already in dict - assign i2 to the same poi
        elif i1 in id_to_poi:
            thresh = threshold_big if poi_counts[id_to_poi[i1]] > 2 else threshold_small
            if pred > thresh:
                id_to_poi[i2] = id_to_poi[i1]
                poi_to_id[id_to_poi[i1]].append(i2)
                poi_counts[id_to_poi[i1]] += 1

        # i2 is already in dict - assign i1 to the same poi
        elif i2 in id_to_poi:
            thresh = threshold_big if poi_counts[id_to_poi[i2]] > 2 else threshold_small
            if pred > thresh:
                id_to_poi[i1] = id_to_poi[i2]
                poi_to_id[id_to_poi[i2]].append(i1)
                poi_counts[id_to_poi[i2]] += 1

        # New POI
        else:
            id_to_poi[i1] = poi
            id_to_poi[i2] = poi

            poi_to_id[poi] = [i1, i2]

            poi_counts[poi] = 2
            poi += 1

    return id_to_poi, poi_to_id, poi_counts


def get_poi_to_id(id_to_poi):
    poi_to_id = {}
    for k in id_to_poi.keys():
        if id_to_poi[k] not in poi_to_id:
            poi_to_id[id_to_poi[k]] = []
        poi_to_id[id_to_poi[k]].append(k)

    return poi_to_id


def merge_pois_simple(
    df2, id_to_poi, poi_to_id, poi_counts, threshold=0.5, threshold_merge=0.8, max_size=300
):
    matches = df2[df2["match"] > threshold].reset_index(drop=True)

    id1, id2, preds = np.split(matches.values, [1, 2], axis=1)
    id1, id2, preds = id1.flatten(), id2.flatten(), preds.flatten()

    for i, (i1, i2, pred) in enumerate(zip(id1, id2, preds)):
        if i1 in id_to_poi and i2 in id_to_poi:
            # only merge if combined size is <= 300 and pred > th3
            if (
                id_to_poi[i2] != id_to_poi[i1]
                and pred > threshold_merge
                and poi_counts[id_to_poi[i1]] + poi_counts[id_to_poi[i2]] <= max_size
            ):
                m = min(id_to_poi[i2], id_to_poi[i1])
                m2 = max(id_to_poi[i2], id_to_poi[i1])

                poi_counts[m] = poi_counts[id_to_poi[i1]] + poi_counts[id_to_poi[i2]]
                poi_counts[m2] = 0

                for j in poi_to_id[m2]:
                    id_to_poi[j] = m

                poi_to_id[m] = poi_to_id[m] + poi_to_id[m2]
                poi_to_id[m2] = []

    return id_to_poi, poi_to_id, poi_counts


def merge_pois_advanced(
    df2,
    id_to_poi,
    poi_to_id,
    poi_counts,
    threshold_merge_avg=0.5,
    threshold_merge_max=0.8,
    max_size=300
):
    id1, id2, preds = np.split(df2.values, [1, 2], axis=1)
    id1, id2, preds = id1.flatten(), id2.flatten(), preds.flatten()
    merging_pairs = []

    for i, (i1, i2, pred) in enumerate(zip(id1, id2, preds)):
        if i1 in id_to_poi and i2 in id_to_poi:
            poi1 = id_to_poi[i1]
            poi2 = id_to_poi[i2]

            if id_to_poi[i2] == id_to_poi[i1]:
                continue

            if poi_counts[poi1] + poi_counts[poi2] > 300:
                continue  # Too big, skip

            m = min(poi1, poi2)
            m2 = max(poi2, poi1)
            to_merge = [m, m2, i1, i2, pred]
            merging_pairs.append(to_merge)

    df_merge = pd.DataFrame(merging_pairs, columns=["poi1", "poi2", "i1", "i2", "score"])
    df_merge['poi1_poi2'] = df_merge['poi1'].astype(str) + "_" + df_merge['poi2'].astype(str)

    to_merge = {}

    for pois, merge in df_merge.groupby('poi1_poi2'):

        m, m2, i1, i2 = merge[['poi1', 'poi2', 'i1', 'i2']].values[0]

    #     s1 = poi_counts[id_to_poi[i1]]
    #     s2 = poi_counts[id_to_poi[i2]]
    #     links_prop = len(merge) / min(s1, s2)

        if (
            (merge['score'].max() > threshold_merge_max) or
            (merge['score'].mean() > threshold_merge_avg and len(merge) > 1)
            # or (links_prop > 0.25):
        ):
            try:
                to_merge[m2] = to_merge[m]
            except KeyError:
                to_merge[m2] = m

    for m2, m in to_merge.items():
        if poi_counts[m] + poi_counts[m2] > max_size:
            continue

        poi_counts[m] = poi_counts[m] + poi_counts[m2]
        poi_counts[m2] = 0

        for poi in poi_to_id[m2]:
            id_to_poi[poi] = m

        poi_to_id[m] = poi_to_id[m] + poi_to_id[m2]
        poi_to_id[m2] = []

    return id_to_poi, poi_to_id, poi_counts


def iou(preds, truths):
    p = set(preds.strip().split(' '))
    t = set(truths.strip().split(' '))
    return len(p & t) / len(p | t)


def evaluate(df, m2):
    # bring predictions into final result
    p1_tr = df[["id", "m_true"]].merge(m2[["id", "m2"]], on="id", how="left")

    # fill missing values with self-match
    p1_tr["m2"] = p1_tr["m2"].fillna("missing")
    idx = p1_tr["m2"] == "missing"
    p1_tr["m2"].loc[idx] = (
        p1_tr["id"].astype("str").loc[idx]
    )

    # Compute metric
    x1, x2 = p1_tr["m_true"].to_numpy(), p1_tr["m2"].to_numpy()
    p1_tr["ious"] = [iou(pred, truth) for pred, truth in zip(x2, x1)]

    cv = (p1_tr["ious"]).mean()
    return cv


def remove_outliers(df2, id_to_poi, poi_to_id, threshold_out=0.25):
    id1, id2, preds = np.split(df2.values, [1, 2], axis=1)
    id1, id2, preds = id1.flatten(), id2.flatten(), preds.flatten()

    poi_to_pair_idx = {}

    for i, (i1, i2, pred) in enumerate(zip(id1, id2, preds)):
        poi1 = id_to_poi.get(i1, -1)
        poi2 = id_to_poi.get(i2, -1)
        if poi1 > -1 and poi2 > -1 and poi1 == poi2:
            try:
                poi_to_pair_idx[poi1].add(i)
            except Exception:
                poi_to_pair_idx[poi1] = set([i])

    # filter out
    current_clust = np.max(list(id_to_poi.values())) + 1

    for clust in poi_to_id:
        if len(poi_to_id[clust]) < 3:
            continue

        df_clust = df2.iloc[list(poi_to_pair_idx[clust])]

        for id_ in poi_to_id[clust]:
            scores = df_clust.query(f'id == "{id_}" | id2 == "{id_}"')['match']

            if (
                scores.mean() < threshold_out
            ):
                id_to_poi[id_] = current_clust
                current_clust += 1

    poi_to_id = get_poi_to_id(id_to_poi)

    return id_to_poi, poi_to_id


def get_improved_CV(df_p, pred_oof, df_gt, thresholds=[0.45, 0.6, 0.6, 0.9, 0], max_size=300):
    threshold, threshold_small, threshold_big, threshold_merge_max, threshold_merge_avg = thresholds
    df2 = df_p.copy()
    df2["match"] = np.copy(pred_oof)

    try:
        df2 = df2[['id_1', 'id_2', "match"]]
        df2.columns = ['id', 'id2', "match"]
    except KeyError:
        df2 = df2[['id', 'id2', "match"]]

    # sort by decr prediction
    df2 = df2.sort_values(by=["match"], ascending=False).reset_index(drop=True)

    # Build clusters
    id_to_poi, poi_to_id, poi_counts = match_pois(
        df2,
        threshold=threshold,
        threshold_small=threshold_small,
        threshold_big=threshold_big
    )

    # Merge clusters
    if threshold_merge_avg > 0:
        id_to_poi, poi_to_id, poi_counts = merge_pois_advanced(
            df2,
            id_to_poi,
            poi_to_id,
            poi_counts,
            threshold_merge_avg=threshold_merge_avg,
            threshold_merge_max=threshold_merge_max,
            max_size=max_size
        )
    else:
        id_to_poi, poi_to_id, poi_counts = merge_pois_simple(
            df2,
            id_to_poi,
            poi_to_id,
            poi_counts,
            threshold=threshold,
            threshold_merge=threshold_merge_max,
            max_size=max_size
        )

    # Reformat
    preds = pd.DataFrame.from_dict(id_to_poi, orient="index").reset_index()
    preds['matches'] = preds[0].map(poi_to_id).apply(lambda x: " ".join(x))
    preds.columns = ["id", "poi", "m2"]

    # Score
    cv = evaluate(df_gt, preds)
    print(f"CV {cv:.4f}")

    return id_to_poi, poi_to_id, poi_counts, preds, cv
