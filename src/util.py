import ast
import numpy as np
import pandas as pd


TYPES = {
    "id": np.dtype("O"),
    "name": np.dtype("O"),
    "latitude": np.dtype("float32"),
    "longitude": np.dtype("float32"),
    "address": np.dtype("O"),
    "city": np.dtype("O"),
    "state": np.dtype("O"),
    "zip": np.dtype("O"),
    "country": np.dtype("int16"),
    "url": np.dtype("O"),
    "phone": np.dtype("O"),
    "categories": np.dtype("O"),
    "point_of_interest": np.dtype("int32"),
    "lang": np.dtype("int8"),
    "m_true": np.dtype("O"),
    "category_simpl": np.dtype("int16"),
    "categories_split": np.dtype("O"),
    "name_initial": np.dtype("O"),
    "name_initial_decode": np.dtype("O"),
    "freq_pairing_with_other_groupedcat": np.dtype("float32"),
    "cat_solo_score": np.dtype("float32"),
    "Nb_multiPoi": np.dtype("float32"),
    "mean": np.dtype("float32"),
    "q25": np.dtype("float32"),
    "q50": np.dtype("float32"),
    "q75": np.dtype("float32"),
    "q90": np.dtype("float32"),
    "q99": np.dtype("float32"),
    "nameC": np.dtype("O"),
    "lat2": np.dtype("float32"),
    "lon2": np.dtype("float32"),
    "name2": np.dtype("O"),
    "cat2": np.dtype("int16"),
}


def load_cleaned_data(path):
    df = pd.read_csv(path, dtype=TYPES)
    # df = pd.read_csv(OUT_PATH + "TRAIN_CHECK.csv", dtype=TYPES)

    df.fillna("", inplace=True)

    df["categories_split"] = df["categories_split"].apply(
        ast.literal_eval
    )

    return df
