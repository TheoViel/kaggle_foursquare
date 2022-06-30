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
    df2, id_to_poi, poi_to_id, poi_counts, threshold=0.5, threshold_merge=0.8
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
                and poi_counts[id_to_poi[i1]] + poi_counts[id_to_poi[i2]] <= 300
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


def get_improved_CV(df_p, pred_oof, df_gt, thresholds=[0.45, 0.6, 0.6, 0.9]):
    threshold, threshold_small, threshold_big, threshold_merge = thresholds
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
    id_to_poi, poi_to_id, poi_counts = merge_pois_simple(
        df2,
        id_to_poi,
        poi_to_id,
        poi_counts,
        threshold=threshold,
        threshold_merge=threshold_merge,
    )

    # Reformat
    preds = pd.DataFrame.from_dict(id_to_poi, orient="index").reset_index()
    preds['matches'] = preds[0].map(poi_to_id).apply(lambda x: " ".join(x))
    preds.columns = ["id", "poi", "m2"]

    # Score
    cv = evaluate(df_gt, preds)
    print(f"CV {cv:.4f}")

    return cv
