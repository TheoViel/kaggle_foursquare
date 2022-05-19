# import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.defice = device

        self.loss = nn.TripletMarginLoss(margin=config["margin"], p=config["p"], reduction="none")

    def forward(self, pred):

        anchor, pos, neg = pred.view(3, -1, pred.size(-1))

        return self.loss(anchor, pos, neg)
