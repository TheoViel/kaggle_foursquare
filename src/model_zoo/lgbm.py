from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


def objective_lgbm(trial, df_train, df_val, features, target):
    lgbm_params = {
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
    }

    model = LGBMClassifier(
        **lgbm_params, boosting_type="gbdt", objective="binary", random_state=42
    )

    model.fit(df_train[features], df_train[target])

    pred = model.predict_proba(df_val[features])[:, 1]

    loss = roc_auc_score(df_val[target].values, pred)

    return loss


def train_lgbm(df_train, df_val, df_test, features, target, params=None, i=0):

    model = LGBMClassifier(
        **params, boosting_type="gbdt", objective="binary", random_state=42 + i
    )

    model.fit(df_train[features], df_train[target])

    pred = model.predict_proba(df_val[features])[:, 1]

    if df_test is not None:
        pred_test = model.predict_proba(df_test[features])[:, 1]
    else:
        pred_test = None

    return pred, pred_test, model.feature_importances_
