# import lofo
# import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
# from sklearn.metrics import roc_auc_score


def train_lgbm(
    df_train,
    df_val,
    df_test,
    features,
    target="match",
    params=None,
    cat_features=[],
    i=0,
):

    model = LGBMClassifier(
        **params,
        n_estimators=10000,
        objective="binary",
        learning_rate=0.05,
        device="gpu",
        random_state=42 + i,
    )

    model.fit(
        df_train[features],
        df_train[target],
        eval_set=[(df_val[features], df_val[target])],
        eval_metric="auc",
        callbacks=[early_stopping(100), log_evaluation(100)],
        # early_stopping_rounds=100,  # None
        # categorical_feature=cat_features,
    )

    pred = model.predict_proba(df_val[features])[:, 1]

    return pred, model
