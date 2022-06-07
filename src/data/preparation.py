import re
import ast
import pykakasi
import numpy as np
import pandas as pd
from urllib.parse import urlparse


def get_full_address(x):
    string = ""
    string = string + x.address + ", " if x.address else string
    string = string + x.city + ", " if x.city else string
    string = string + x.state + ", " if x.state else string
    string = string + x.zip + " " if x.zip else string
    string = string + x.country if x.country else string

    return string.strip()


def parse_url(url):
    string = ""
    if len(url) == 0:
        return ""

    result = urlparse(url)
    string += re.sub("www.", "", result.netloc) + " "

    if len(result.path):
        path = result.path.rsplit('.', 1)[0].lower()
        string += re.sub('[^a-z]+', " ", path).strip()

    return string.strip()


def prepare_train_data(root=""):
    """
    Prepares the metadata.

    Args:
        root (str, optional): Path to metadata. Defaults to "".

    Returns:
        pandas DataFrame: Prepared metadata.
    """
    df = pd.read_csv(root + "df_train.csv")

    cols = ['name', 'address', 'city', 'state', 'zip', 'country', 'phone', 'url', 'categories']
    df[cols] = df[cols].fillna('').astype(str)

    df['full_address'] = df.apply(get_full_address, 1)
    # df.drop(['city', 'state', 'zip'], axis=1, inplace=True)

    df['url'] = df['url'].apply(parse_url)

    return df.set_index('id')


def prepare_nn_data(df):
    """
    Prepares the metadata.

    Args:
        root (str, optional): Path to metadata. Defaults to "".

    Returns:
        pandas DataFrame: Prepared metadata.
    """
    df = df.to_pandas().reset_index()
    cols = ['name', 'address', 'city', 'state', 'zip', 'country', 'phone', 'url', 'categories']
    df[cols] = df[cols].fillna('').astype(str)

    df['full_address'] = df.apply(get_full_address, 1)
    # df.drop(['city', 'state', 'zip'], axis=1, inplace=True)

    return df.set_index('id')


def prepare_triplet_data(root=""):
    """
    Prepares the triplets.

    Args:
        root (str, optional): Path to metadata. Defaults to "".

    Returns:
        pandas DataFrame: Prepared metadata.
    """
    triplets = pd.read_csv(root + 'triplets.csv')  # _v2

    triplets['paired_ids'] = triplets['paired_ids'].apply(ast.literal_eval)
    triplets['matches'] = triplets['matches'].apply(ast.literal_eval)

    # triplets.drop('fp_ids', axis=1, inplace=True)
    triplets['fp_ids'] = triplets['fp_ids'].apply(lambda x: x.split(' '))

    triplets['pos_ids'] = triplets.apply(
        lambda x: [i for i, t in zip(x.paired_ids, x.matches) if t], 1
    )
    triplets['neg_ids'] = triplets.apply(
        lambda x: [i for i, t in zip(x.paired_ids, x.matches) if not t], 1
    )

    return triplets


def reduce_mem_usage(df, verbose=1):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def convert_japanese_alphabet(df: pd.DataFrame):
    kakasi = pykakasi.kakasi()
    kakasi.setMode('H', 'a')  # Convert Hiragana into alphabet
    kakasi.setMode('K', 'a')  # Convert Katakana into alphabet
    kakasi.setMode('J', 'a')  # Convert Kanji into alphabet
    conversion = kakasi.getConverter()

    def convert(row):
        for column in ["name", "address", "city", "state"]:
            if isinstance(row[column], str):
                row[column] = conversion.do(row[column])
            # except KeyboardInterrupt():
            #     pass
        return row

    df[df["country"] == "JP"] = df[df["country"] == "JP"].parallel_apply(convert, axis=1)
    return df
