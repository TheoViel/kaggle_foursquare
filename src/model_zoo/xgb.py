import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def objective_xgb(trial, df_train, df_val, features, target="match"):
    xgb_params = dict(
        max_depth=trial.suggest_int("max_depth", 5, 15),
        # max_leaves=trial.suggest_int("max_leaves", 100, 10000),
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
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="gpu_hist",
        predictor='gpu_predictor',
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

    y_val = df_val[target].values if isinstance(df_val, pd.DataFrame) else df_val[target].get()

    return roc_auc_score(y_val, pred)


def train_xgb(df_train, df_val, df_test, features, target="match", params=None, i=0):

    model = XGBClassifier(
        **params,
        n_estimators=10000,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="gpu_hist",
        predictor='gpu_predictor',
        use_label_encoder=False,
        random_state=42 + i,
    )

    model.fit(
        df_train[features],
        df_train[target],
        eval_set=[(df_val[features], df_val[target])],
        verbose=100,
        early_stopping_rounds=50,
    )

    pred = model.predict_proba(df_val[features])[:, 1]

    return pred, model
