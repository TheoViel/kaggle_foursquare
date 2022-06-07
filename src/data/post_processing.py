import copy
import cudf
import numpy as np
import pandas as pd


def limit_numbers(preds, scores, n=2):
    preds_pp = copy.deepcopy(preds)
    for k in preds:
        if len(preds[k]) > n:
            order = np.argsort(scores[k])
            preds_pp[k] = list(np.array(preds[k])[order[:n]])

    return preds_pp


def post_process_matches(matches, mode="append"):
    new_matches = copy.deepcopy(matches)
    for k in matches:
        for m in matches[k]:
            if k not in new_matches[m]:
                if mode == "remove":
                    new_matches[k].remove(m)
                elif mode == "append":
                    new_matches[m].append(k)
                else:
                    raise NotImplementedError

    return new_matches


def preds_to_matches(preds, df, threshold=0.5, ids=None):
    gpu = not isinstance(df, pd.DataFrame)

    if ids is None:
        ids = df['id_1'].unique()

    identity = pd.DataFrame({"id_1": ids, "id_2": ids, "pred": 1})

    df['pred'] = preds
    df = df[df['pred'] > threshold].reset_index(drop=True)
    df = df[['id_1', 'id_2', 'pred']].reset_index(drop=True)

    if gpu:
        df = cudf.concat([df, cudf.from_pandas(identity)])
        dfg = df.groupby('id_1').agg(list).to_pandas()
    else:
        df = pd.concat([df, identity])
        dfg = df.groupby('id_1').agg(list)

    dfg['id_2'] = dfg['id_2'].apply(list)
    dfg['pred'] = dfg['pred'].apply(list)
    dfg = dfg.to_dict()
    return dfg['id_2'], dfg['pred']
