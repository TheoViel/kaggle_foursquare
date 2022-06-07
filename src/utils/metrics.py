import cudf
import json
import numpy as np
import pandas as pd
from params import DATA_PATH


def get_ids(root="", fold=None):
    df = cudf.read_csv(root + "df_train.csv")
    folds = cudf.read_csv(DATA_PATH + "folds_2.csv")[['id', 'fold']]
    df = df.merge(folds, how="left", on="id").set_index("id")

    if fold is not None:
        df = df[df['fold'] == fold]

    return list(df.index.to_pandas())


def get_id2poi(input_df: pd.DataFrame) -> dict:
    return dict(zip(input_df['id'], input_df['point_of_interest']))


def get_poi2ids(input_df: pd.DataFrame) -> dict:
    return input_df.groupby('point_of_interest')['id'].apply(set).to_dict()


def build_gt(df, save=True):
    id2poi = get_id2poi(df)
    poi2ids = get_poi2ids(df)

    gts = {}
    for id_ in df['id']:
        gts[id_] = list(poi2ids[id2poi[id_]])

    if save:
        with open(DATA_PATH + "gt.json", "w") as f:
            json.dump(gts, f)

    return gts


def compute_iou(preds, gt=None, ids=None):
    if gt is None:
        gt = json.load(open(DATA_PATH + "gt.json", 'r'))
    ious = []

    if ids is None:
        ids = preds.keys()

    for id_ in ids:
        tgt = set(gt[id_])
        pred = set(preds.get(id_, [id_]))

        iou = len((tgt & pred)) / len((tgt | pred))
        ious.append(iou)

    return np.mean(ious)


def compute_best_iou(preds, gt=None):
    ious = []
    for id_ in preds.keys():
        tgt = set(gt[id_])
        pred = preds.get(id_, [])

        iou = len(tgt & set(pred + [id_])) / len(tgt)
        ious.append(iou)

    return np.mean(ious)


def compute_found_prop(preds, gt=None, ids=None):
    if gt is None:
        gt = json.load(open(DATA_PATH + "gt.json", 'r'))

    found_props = []
    missed = []

    if ids is None:
        ids = preds.keys()

    for id_ in ids:
        if len(gt[id_]) == 1:
            missed.append([])
            continue

        tgt = set([g for g in gt[id_] if g != id_])
        pred = set(preds.get(id_, [id_]))

        found_prop = len((tgt & pred)) / len(tgt)
        found_props.append(found_prop)

        missed.append(tgt - pred)

        # break

    return np.mean(found_props), missed
