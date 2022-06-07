import gc
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from training.train_nli import fit
from model_zoo.models import NLITransformer

from data.dataset import NLIDataset
from data.tokenization import get_tokenizer
from utils.torch import seed_everything, count_parameters, save_model_weights


def train(config, tokenizer, df_train, df_val, fold, log_folder=None):
    """
    Trains and validate a model.
    TODO

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        np array [len(df_train) x num_classes]: Validation predictions.
        np array [len(df_train) x num_classes_aux]: Validation auxiliary predictions.
    """
    seed_everything(config.seed)

    train_dataset = NLIDataset(
        df_train,
        tokenizer,
        features=config.features,
        max_len=config.max_len,
        train=True,
    )

    val_dataset = NLIDataset(
        df_val,
        tokenizer,
        features=config.features,
        max_len=config.max_len,
    )

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(".pt") or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = glob.glob(config.pretrained_weights + f"*_{fold}.pt")[0]
    else:
        pretrained_weights = None

    model = NLITransformer(
        config.name,
        nb_layers=config.nb_layers,
        no_dropout=config.no_dropout,
        num_classes=config.num_classes,
        nb_features=config.nb_features,
        d=config.d,
        pretrained_weights=pretrained_weights
    ).cuda()
    model.zero_grad()
    model.train()

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training pairs")
    print(f"    -> {len(val_dataset)} validation pairs")
    print(f"    -> {n_parameters} trainable parameters\n")

    preds = fit(
        model,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        epochs=config.epochs,
        acc_steps=config.acc_steps,
        verbose_eval=config.verbose_eval,
        device=config.device,
        use_fp16=config.use_fp16,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    if log_folder is not None:
        np.save(log_folder + f"pred_val_{fold}.npy", preds)
        save_model_weights(
            model,
            f"{config.name.split('/')[-1]}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, train_dataset, val_dataset)
    torch.cuda.empty_cache()
    gc.collect()

    return preds


def k_fold(config, df, df_extra=None, log_folder=None):
    tokenizer = get_tokenizer(config.name)

    folds = pd.read_csv(config.folds_file)[['id', 'fold']]

    df = df.merge(folds[['id', 'fold']], how="left", left_on="id_1", right_on="id")
    df.drop('id', axis=1, inplace=True)
    df = df.merge(
        folds[['id', 'fold']], how="left", left_on="id_2", right_on="id", suffixes=("_1", "_2")
    )
    df.drop('id', axis=1, inplace=True)

    pred_oof = np.zeros(len(df))

    for fold in range(config.k):
        if fold in config.selected_folds:
            print(f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n")
            seed_everything(fold)

            df_train = df[(df['fold_1'] != fold) & (df['fold_2'] != fold)].reset_index(drop=True)
            df_val = df[(df['fold_1'] == fold) | (df['fold_2'] == fold)]
            val_idx = df_val.index.values

            pred_val = train(
                config,
                tokenizer,
                df_train.reset_index(),
                df_val.reset_index(),
                fold,
                log_folder=log_folder
            )

            if log_folder is None:
                return pred_val
            pred_oof[val_idx] = pred_val

    if config.selected_folds == list(range(config.k)):
        score = roc_auc_score(df['match'].values, pred_oof)
        print(f"\n\n -> CV score : {score:.4f}")
        if log_folder is not None:
            np.save(log_folder + "pred_oof.npy", pred_oof)

    return pred_oof
