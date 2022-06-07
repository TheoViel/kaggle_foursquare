import torch
import transformers
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# from utils.torch import load_pretrained_weights


class SingleTransformer(nn.Module):
    def __init__(
        self,
        model,
        nb_layers=1,
        embed_dim=128,
        nb_features=0,
        config_file=None,
        pretrained=True,
        pretrained_weights=None,
        no_dropout=False,
    ):
        super().__init__()
        self.name = model
        self.nb_layers = nb_layers
        self.nb_features = nb_features

        self.pad_idx = 1 if "roberta" in self.name else 0

        transformers.logging.set_verbosity_error()

        if config_file is None:
            config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        else:
            config = torch.load(config_file)

        if no_dropout:
            config.hidden_dropout_prob = 0
            config.attention_probs_dropout_prob = 0

        if pretrained:
            self.transformer = AutoModel.from_pretrained(model, config=config)
        else:
            self.transformer = AutoModel.from_config(config)

        in_fts = config.hidden_size * self.nb_layers + nb_features

        self.dense = nn.Sequential(
            nn.Linear(in_fts, embed_dim),
            nn.Tanh(),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)
        )

        if pretrained_weights is not None:
            raise NotImplementedError()
            # load_pretrained_weights(self, pretrained_weights)

    def forward(self, tokens, token_type_ids=None, fts=None):
        """
        Usual torch forward function

        Arguments:
            tokens {torch tensor} -- Sentence tokens
            token_type_ids {torch tensor} -- Sentence tokens ids
        """
        token_type_ids = torch.zeros(
            tokens.size(), dtype=torch.long, device=tokens.device
        ) if token_type_ids is None else token_type_ids

        hidden_states = self.transformer(
            tokens,
            attention_mask=(tokens != self.pad_idx).long(),
            token_type_ids=token_type_ids,
        )[-1]

        hidden_states = hidden_states[::-1]
        features = torch.cat(hidden_states[:self.nb_layers], -1)[:, 0]

        if fts is not None and self.nb_features > 0:
            features = torch.cat([features, fts], -1)

        representation = self.dense(features)
        projection = self.projection_head(representation)

        return representation, projection


class NLITransformer(nn.Module):
    def __init__(
        self,
        model,
        nb_layers=1,
        d=256,
        num_classes=128,
        nb_features=0,
        config_file=None,
        pretrained=True,
        pretrained_weights=None,
        no_dropout=False,
    ):
        super().__init__()
        self.name = model
        self.nb_layers = nb_layers
        self.nb_features = nb_features

        self.pad_idx = 1 if "roberta" in self.name else 0

        transformers.logging.set_verbosity_error()

        if config_file is None:
            config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        else:
            config = torch.load(config_file)

        if no_dropout:
            config.hidden_dropout_prob = 0
            config.attention_probs_dropout_prob = 0

        if pretrained:
            self.transformer = AutoModel.from_pretrained(model, config=config)
        else:
            self.transformer = AutoModel.from_config(config)

        in_fts = config.hidden_size * self.nb_layers + nb_features

        self.logits = nn.Sequential(
            nn.Linear(in_fts, d),
            nn.Tanh(),
            nn.Linear(d, num_classes)
        )

        if pretrained_weights is not None:
            raise NotImplementedError()
            # load_pretrained_weights(self, pretrained_weights)

    def forward(self, tokens, token_type_ids=None, fts=None):
        """
        Usual torch forward function

        Arguments:
            tokens {torch tensor} -- Sentence tokens
            token_type_ids {torch tensor} -- Sentence tokens ids
        """
        token_type_ids = torch.zeros(
            tokens.size(), dtype=torch.long, device=tokens.device
        ) if token_type_ids is None else token_type_ids

        hidden_states = self.transformer(
            tokens,
            attention_mask=(tokens != self.pad_idx).long(),
            token_type_ids=token_type_ids,
        )[-1]

        hidden_states = hidden_states[::-1]
        features = torch.cat(hidden_states[:self.nb_layers], -1).mean(1)

        if fts is not None and self.nb_features > 0:
            features = torch.cat([features, fts], -1)

        logits = self.logits(features)
        return logits
