import gc
import glob
import torch
import pandas as pd

from training.train import fit
from model_zoo.models import SingleTransformer

from data.dataset import TripletDataset
from data.tokenization import get_tokenizer
from utils.torch import seed_everything, count_parameters, save_model_weights


def train(config, tokenizer, df_train, df_val, triplets_train, triplets_val, fold, log_folder=None):
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

    train_dataset = TripletDataset(
        df_train,
        triplets_train,
        tokenizer,
        max_len=config.max_len,
        use_name=config.use_name,
        use_address=config.use_address,
        use_url=config.use_url,
        train=True,
    )

    val_dataset = TripletDataset(
        df_val,
        triplets_val,
        tokenizer,
        use_name=config.use_name,
        use_address=config.use_address,
        use_url=config.use_url,
        max_len=config.max_len,
    )

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(".pt") or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = glob.glob(config.pretrained_weights + f"*_{fold}.pt")[0]
    else:
        pretrained_weights = None

    model = SingleTransformer(
        config.name,
        nb_layers=config.nb_layers,
        no_dropout=config.no_dropout,
        embed_dim=config.embed_dim,
        nb_features=config.nb_features,
        pretrained_weights=pretrained_weights
    ).cuda()
    model.zero_grad()
    model.train()

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training triplets")
    print(f"    -> {len(val_dataset)} validation triplets")
    print(f"    -> {n_parameters} trainable parameters\n")

    fit(
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
        save_model_weights(
            model,
            f"{config.name.split('/')[-1]}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, train_dataset, val_dataset)
    torch.cuda.empty_cache()
    gc.collect()


def k_fold(config, df, triplets, df_extra=None, log_folder=None):
    tokenizer = get_tokenizer(config.name)

    folds = pd.read_csv(config.folds_file)[['id', 'fold']]
    triplets = triplets.merge(folds, how="left", on="id")
    df = df.merge(folds, how="left", on="id").set_index("id")

    for fold in range(config.k):
        if fold in config.selected_folds:
            print(f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n")

            seed_everything(fold)

            triplets_train = triplets[triplets['fold'] != fold].reset_index(drop=True)
            triplets_val = triplets[triplets['fold'] == fold]
            triplets_val = triplets_val.sample(len(triplets_val) // 10).reset_index(drop=True)

            df_train = df[df['fold'] != fold]
            df_val = df[df['fold'] == fold]

            train(
                config,
                tokenizer,
                df_train,
                df_val,
                triplets_train,
                triplets_val,
                fold,
                log_folder=log_folder
            )
