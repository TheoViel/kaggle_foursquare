import json
import numpy as np
import pandas as pd
from cuml import ForestInference
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from utils.logger import Config


def k_fold_inf(
    df,
    log_folder,
):
    config = Config(json.load(open(log_folder + "config.json", "r")))
    pred_oof = np.zeros(len(df))

    if config.split == "kf":
        kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=13)
        splits = kf.split(df)
    elif config.split == "gkf":
        splits = [(i, i) for i in range(config.n_folds)]
    else:
        raise NotImplementedError()

    for fold, (train_idx, val_idx) in enumerate(splits):
        if fold in config.selected_folds:
            print(f"\n-------------   Fold {fold + 1} / {config.n_folds}  -------------\n")

            if config.split == "kf":
                df_val = df.iloc[val_idx].reset_index(drop=True)
            else:
                df_val = df[(df["fold_1"] == fold) | (df["fold_2"] == fold)]

                val_idx = (
                    df_val.index.values
                    if isinstance(df, pd.DataFrame)
                    else df_val.index.values.get()
                )
            print(f"- Scoring {len(df_val)} pairs\n")

            if config.model == "lgbm":
                model = ForestInference.load(log_folder + f"lgbm_{fold}.txt", model_type="lightgbm")
                pred_val = model.predict(df_val[config.features]).flatten()

            elif config.model == "xgb":
                model = ForestInference.load(
                    log_folder + f"xgb_{fold}.json", output_class=True, model_type="xgboost_json"
                )
                # model = xgboost.XGBClassifier()
                # model.load_model(log_folder + f"xgb_{fold}.json")

                pred_val = model.predict_proba(df_val[config.features])[:, 1]
            else:
                raise NotImplementedError

            print(f"- AUC = {roc_auc_score(df_val[config.target], pred_val):.4f}")
            pred_oof[val_idx] += pred_val

    pred_oof = pred_oof / (1 + (df["fold_1"] != df["fold_2"]))

    print(f"\n -> CV AUC = {roc_auc_score(df[config.target], pred_oof) :.4f}\n")

    return pred_oof
