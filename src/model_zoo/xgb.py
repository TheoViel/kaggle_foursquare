import lofo
import pandas as pd
from xgboost import XGBClassifier, callback
from sklearn.metrics import roc_auc_score


def objective_xgb(trial, df_train, df_val, features, target="match"):
    xgb_params = dict(
        max_depth=trial.suggest_int("max_depth", 5, 15),
        gamma=trial.suggest_float("gamma", 1e-6, 1e-1, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1),
        subsample=trial.suggest_float("subsample", 0.5, 1),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1, log=True),
    )

    model = XGBClassifier(
        **xgb_params,
        n_estimators=10000,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        use_label_encoder=False,
    )

    model.fit(
        df_train[features],
        df_train[target],
        eval_set=[(df_val[features], df_val[target])],
        verbose=0,
        early_stopping_rounds=20,
    )

    pred = model.predict_proba(df_val[features])[:, 1]

    y_val = (
        df_val[target].values
        if isinstance(df_val, pd.DataFrame)
        else df_val[target].get()
    )

    return roc_auc_score(y_val, pred)


def train_xgb(
    df_train,
    df_val,
    df_test,
    features,
    target="match",
    params=None,
    cat_features=[],
    i=0,
):

    model = XGBClassifier(
        **params,
        n_estimators=10000,
        objective="binary:logistic",
        learning_rate=0.05,
        eval_metric="auc",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        use_label_encoder=False,
        random_state=42 + i,
    )

    es = callback.EarlyStopping(
        rounds=100,
        min_delta=1e-5,
        save_best=True,
        maximize=True,
        data_name="validation_0",
        metric_name="auc",
    )

    model.fit(
        df_train[features],
        df_train[target],
        eval_set=[(df_val[features], df_val[target])],
        verbose=100,
        # early_stopping_rounds=100,  # None
        callbacks=[es],
    )

    pred = model.predict_proba(df_val[features])[:, 1]

    return pred, model


def lofo_xgb(df, config, folds=[0], auto_group_threshold=1):
    dataset = lofo.Dataset(
        df,
        target=config.target,
        features=config.features,
        auto_group_threshold=auto_group_threshold,
    )

    cv = []
    for fold in range(config.n_folds):
        if fold in folds:
            df_train_opt = df[(df["fold_1"] != fold) & (df["fold_2"] != fold)]
            df_val_opt = df[(df["fold_1"] == fold) | (df["fold_2"] == fold)]
            cv.append((list(df_train_opt.index), list(df_val_opt.index)))

    model = XGBClassifier(
        **config.params,
        n_estimators=10000,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        use_label_encoder=False,
    )

    lofo_imp = lofo.LOFOImportance(dataset, scoring="roc_auc", cv=cv, model=model)

    return lofo_imp.get_importance()
