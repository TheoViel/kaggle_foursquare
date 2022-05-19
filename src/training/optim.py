import torch


def define_optimizer(name, params, lr=1e-3, betas=(0.9, 0.999)):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.
    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters
        lr (float, optional): Learning rate. Defaults to 1e-3.
    Raises:
        NotImplementedError: Specified optimizer name is not supported.
    Returns:
        torch optimizer: Optimizer
    """
    try:
        optimizer = getattr(torch.optim, name)(params, lr=lr, betas=betas)
    except AttributeError:
        raise NotImplementedError

    return optimizer


def custom_params(model, weight_decay=0, lr=1e-3, lr_transfo=3e-5, lr_decay=1):
    """
    Custom parameters for Bert Models to handle weight decay and differentiated learning rates.

    Args:
        model (torch model]): Transformer model
        weight_decay (int, optional): Weight decay. Defaults to 0.
        lr (float, optional): LR of layers not belonging to the transformer. Defaults to 1e-3.
        lr_transfo (float, optional): LR of the last layer of the transformer. Defaults to 3e-5.
        lr_decay (float, optional): Factor to multiply lr_transfo when going deeper. Defaults to 1.

    Returns:
        list: Optimizer params.
    """

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    transformer_name = "transformer"
    opt_params = []

    if not any([n in model.name for n in ["albert", "funnel", "distil"]]):
        try:
            nb_blocks = len(model.transformer.encoder.layer)
        except AttributeError:
            nb_blocks = len(model.transformer.encoder.layers)

        for n, p in model.named_parameters():
            wd = 0 if any(nd in n for nd in no_decay) else weight_decay

            if transformer_name in n and "pooler" not in n:
                lr_ = None
                for i in range(nb_blocks):  # for bert base
                    if f"layer.{i}." in n or f"layers.{i}." in n:
                        lr_ = lr_transfo * lr_decay ** (nb_blocks - 1 - i)
                        break

                if lr_ is None:  # embedding related layers
                    # print(n)
                    lr_ = lr_transfo * lr_decay ** (nb_blocks)

            else:
                lr_ = lr

            opt_params.append(
                {
                    "params": [p],
                    "weight_decay": wd,
                    "lr": lr_,
                }
            )
    else:
        for n, p in model.named_parameters():
            wd = 0 if any(nd in n for nd in no_decay) else weight_decay

            if transformer_name in n:
                lr_ = lr_transfo
            else:
                lr_ = lr

            opt_params.append(
                {
                    "params": [p],
                    "weight_decay": wd,
                    "lr": lr_,
                }
            )

    return opt_params


def trim_tensors(to_trim, pad_token=0, min_len=10):
    """
    Trim tensors so that within a batch, padding is shortened.
    This speeds up training for RNNs and Transformers

    Args:
        to_trim (list of torch tensors): Tokens to trim. First element has to be ids.
        model_name (str, optional): [description]. Defaults to 'bert'.
        min_len (int, optional): Minimum trimming size. Defaults to 10.

    Returns:
        list of torch tensors: Trimmed tokens.
    """
    max_len = (to_trim[0] != pad_token).sum(1).max()
    max_len = max(max_len, min_len)
    return [tokens[:, :max_len] for tokens in to_trim]
