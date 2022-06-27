import numpy as np
import pandas as pd
from model_zoo.xgb import train_xgb
from model_zoo.catboost import train_catboost
from model_zoo.lgbm import train_lgbm
from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import KFold

TRAIN_FCTS = {
    "lgbm": train_lgbm,
    "xgb": train_xgb,
    "catboost": train_catboost,
}


def k_fold(
    df,
    config,
    log_folder=None,
):
    train_fct = TRAIN_FCTS[config.model]

    ft_imps, models = [], []
    pred_oof = np.zeros(len(df))

    # kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=13)
    # splits = kf.split(df)
    # for fold, (train_idx, val_idx) in enumerate(splits):

    for fold in range(config.n_folds):
        if fold in config.selected_folds:
            print(f"\n-------------   Fold {fold + 1} / {config.n_folds}  -------------\n")

            # df_train = df.iloc[train_idx].reset_index(drop=True)
            # df_val = df.iloc[val_idx].reset_index(drop=True)

            df_train = df[(df["fold_1"] != fold) & (df["fold_2"] != fold)].reset_index(drop=True)
            df_val = df[(df["fold_1"] == fold) | (df["fold_2"] == fold)]

            val_idx = (
                df_val.index.values if isinstance(df, pd.DataFrame) else df_val.index.values.get()
            )

            print(f"    -> {len(df_train)} training pairs")
            print(f"    -> {len(df_val)} validation pairs\n")

            pred_val, model = train_fct(
                df_train,
                df_val.reset_index(drop=True),
                None,
                config.features,
                config.target,
                params=config.params,
                cat_features=config.cat_features,
            )

            pred_oof[val_idx] = pred_val
            ft_imp = pd.DataFrame(
                pd.Series(model.feature_importances_, index=config.features), columns=["importance"]
            )

            ft_imps.append(ft_imp)
            models.append(model)

            if log_folder is None:
                return pred_oof, models, ft_imp

            try:
                model.save_model(log_folder + f"{config.model}_{fold}.txt")
            except:
                 model.booster_.save_model(log_folder + f"{config.model}_{fold}.txt")

    y = df[config.target].values if isinstance(df, pd.DataFrame) else df[config.target].get()
    auc = roc_auc_score(y, pred_oof)
    print(f"\n Local CV is {auc:.4f}")

    ft_imp = pd.concat(ft_imps, axis=1).mean(1)
    ft_imp.to_csv(log_folder + "ft_imp.csv")
    np.save(log_folder + "pred_oof.npy", pred_oof)

    return pred_oof, models, ft_imp
