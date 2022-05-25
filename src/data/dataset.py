import torch
import numpy as np
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(
        self,
        df,
        triplets,
        tokenizer,
        max_len=100,
        train=False,
    ):
        self.df = df
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train = train

    def encode(self, row):
        # print(row, '\n')

        tokens = [self.tokenizer.special_tokens["cls"]]
        max_len = self.max_len - 3

        tokens += self.tokenizer(
            row["name"] + ", " + row["categories"],
            add_special_tokens=False,
            max_length=2 * max_len // 3,
            truncation=True,
        )["input_ids"]
        tokens += [self.tokenizer.special_tokens["sep"]]

        tokens += self.tokenizer(
            row["full_address"],
            add_special_tokens=False,
            max_length=max_len // 3,
            truncation=True,
        )["input_ids"]
        tokens += [self.tokenizer.special_tokens["sep"]]

        # tokens += self.tokenizer(
        #     row["url"],
        #     add_special_tokens=False,
        #     max_length=self.max_len // 4,
        #     truncation=True,
        # )["input_ids"]
        # tokens += [self.tokenizer.special_tokens["sep"]]

        tokens += [self.tokenizer.special_tokens["pad"]] * (self.max_len - len(tokens))

        return tokens

    def __getitem__(self, idx):
        ref_id = self.triplets["id"][idx]

        if self.train:
            pos_id = np.random.choice(self.triplets["pos_ids"][idx])
            neg_id = np.random.choice(self.triplets["fp_ids"][idx])
        else:
            pos_id = self.triplets["pos_ids"][idx][0]
            neg_id = self.triplets["fp_ids"][idx][0]

        ref_tokens = self.encode(self.df.loc[ref_id])
        pos_tokens = self.encode(self.df.loc[pos_id])
        neg_tokens = self.encode(self.df.loc[neg_id])

        fts = self.df.loc[[ref_id, pos_id, neg_id], ['latitude', 'longitude']].values[None]

        return {
            "ref_ids": torch.tensor(ref_tokens, dtype=torch.long),
            "pos_ids": torch.tensor(pos_tokens, dtype=torch.long),
            "neg_ids": torch.tensor(neg_tokens, dtype=torch.long),
            "fts": torch.tensor(fts, dtype=torch.float),
            # "df": self.df.loc[[ref_id, pos_id, neg_id]],
        }

    def __len__(self):
        return len(self.triplets)


class SingleDataset(TripletDataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len=100,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        id_ = self.df.index[idx]
        tokens = self.encode(self.df.loc[id_])
        fts = self.df.loc[id_, ['latitude', 'longitude']].values.astype(float)

        return {
            "ids": torch.tensor(tokens, dtype=torch.long),
            "fts": torch.tensor(fts, dtype=torch.float),
        }

    def __len__(self):
        return len(self.df)
