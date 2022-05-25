from xgboost import XGBRFClassifier
from sklearn.metrics import roc_auc_score


def objective_xgbrf(trial, df_train, df_val, features, target):
    xgbrf_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 10, 500),  # 500
        max_depth=trial.suggest_int("max_depth", 5, 20),
        # eta=trial.suggest_float("eta", 1e-3, .3),  # lr
        gamma=trial.suggest_float("gamma", 1e-6, 1, log=True),
        subsample=trial.suggest_float("colsample_bytree", 0.1, 0.9),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.9),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.05, 1),
        colsample_bynode=trial.suggest_float("colsample_bynode", 0.05, 1),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
    )

    model = XGBRFClassifier(
        **xgbrf_params,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=1,
    )

    model.fit(df_train[features], df_train[target])

    pred = model.predict_proba(df_val[features])[:, 1]

    loss = roc_auc_score(df_val[target].values, pred)

    return loss


def train_xgbrf(df_train, df_val, df_test, features, target, params=None, i=0):

    model = XGBRFClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42 + i,
    )

    model.fit(df_train[features], df_train[target])

    pred = model.predict_proba(df_val[features])[:, 1]

    if df_test is not None:
        pred_test = model.predict_proba(df_test[features])[:, 1]
    else:
        pred_test = None

    return pred, pred_test, model.feature_importances_
