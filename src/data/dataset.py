import re
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
        use_name=True,
        use_address=True,
        use_url=False,
    ):
        self.df = df
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train = train

        self.use_url = use_url
        self.use_address = use_address
        self.use_name = use_name
        self.n = self.use_url + self.use_address + self.use_name

    def encode(self, row):
        max_len = self.max_len - (self.n + 1)
        max_lens = [
            2 * max_len // (self.n + 1),  # twice longer
            max_len // (self.n + 1),
            max_len // (self.n + 1)
        ]

        tokens = [self.tokenizer.special_tokens["cls"]]

        if self.use_name:
            tokens += self.tokenizer(
                row["name"] + ", " + row["categories"],
                add_special_tokens=False,
                max_length=max_lens[0],
                truncation=True,
            )["input_ids"]
            tokens += [self.tokenizer.special_tokens["sep"]]

        if self.use_address:
            tokens += self.tokenizer(
                row["full_address"],
                add_special_tokens=False,
                max_length=max_lens[1] if self.use_name else max_lens[0],
                truncation=True,
            )["input_ids"]
            tokens += [self.tokenizer.special_tokens["sep"]]

        if self.use_url:
            tokens += self.tokenizer(
                row["url"],
                add_special_tokens=False,
                max_length=max_lens[2],
                truncation=True,
            )["input_ids"]
            tokens += [self.tokenizer.special_tokens["sep"]]

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
        use_url=False,
        use_address=True,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.use_url = use_url
        self.use_address = use_address
        self.use_name = True
        self.n = self.use_url + self.use_address + self.use_name

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


class NLIDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        features=None,
        max_len=100,
        train=False,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.features = features
        self.max_len = max_len
        self.train = train

        self.target = df['match']

    def encode(self, row):
        # print(row, '\n')

        tokens = [self.tokenizer.special_tokens["cls"]]
        max_len = self.max_len - 3

        string_1 = row["name_1"] + ", " + row["categories_1"] + " - " + row["full_address_1"]
        string_1 = re.sub(r'\s+', " ", string_1).strip()

        tokens += self.tokenizer(
            string_1,
            add_special_tokens=False,
            max_length=max_len // 2,
            truncation=True,
        )["input_ids"]
        tokens += [self.tokenizer.special_tokens["sep"]]

        string_2 = row["name_2"] + ", " + row["categories_2"] + " - " + row["full_address_2"]
        string_2 = re.sub(r'\s+', " ", string_2).strip()

        tokens += self.tokenizer(
            string_2,
            add_special_tokens=False,
            max_length=max_len // 2,
            truncation=True,
        )["input_ids"]
        tokens += [self.tokenizer.special_tokens["sep"]]

        tokens += [self.tokenizer.special_tokens["pad"]] * (self.max_len - len(tokens))

        return tokens

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        tokens = self.encode(row)
        fts = row[self.features]

        return {
            "ids": torch.tensor(tokens, dtype=torch.long),
            "fts": torch.tensor(fts, dtype=torch.float),
            "target": torch.tensor(self.target[idx], dtype=torch.float),
        }

    def __len__(self):
        return len(self.df)
