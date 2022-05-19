import re
import ast
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

    df['address'] = df.apply(get_full_address, 1)
    # df.drop(['city', 'state', 'zip'], axis=1, inplace=True)

    df['url'] = df['url'].apply(parse_url)

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
