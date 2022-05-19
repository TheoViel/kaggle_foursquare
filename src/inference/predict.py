import torch
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

from params import NUM_WORKERS
from training.optim import trim_tensors


def predict(model, dataset, data_config):
    """
    Usual predict torch function
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=data_config["val_bs"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    preds = []
    with torch.no_grad():
        for data in tqdm(loader):
            ids, = trim_tensors([data['ids']], pad_token=data_config["pad_token"])
            fts = data['fts'].transpose(0, 1).reshape(-1, data['fts'].size(-1))

            pred, _ = model(
                ids.cuda(),
                # token_type_ids.cuda(),
                fts=fts.cuda()
            )
            preds.append(pred.detach().cpu().numpy())

    return np.concatenate(preds)
