import os
import re
import sys
import json
import shutil
import datetime
import subprocess
import numpy as np


class Config:
    """
    Placeholder to load a config from a saved json
    """
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


class Logger(object):
    """
    Simple logger that saves what is printed in a file
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file

    Args:
        directory (str, optional): Path to save logs at. Defaults to "".
        name (str, optional): Name of the file to save the logs in. Defaults to "logs.txt".
    """
    log = open(directory + name, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger


def prepare_log_folder(log_path):
    """
    Creates the directory for logging.
    Logs will be saved at log_path/date_of_day/exp_id

    Args:
        log_path (str): Directory

    Returns:
        str: Path to the created log folder
    """
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_today):
        os.mkdir(log_today)

    exp_id = (
        np.max([int(f) for f in os.listdir(log_today)]) + 1
        if len(os.listdir(log_today))
        else 0
    )
    log_folder = log_today + f"{exp_id}/"

    assert not os.path.exists(log_folder), "Experiment already exists"
    os.mkdir(log_folder)

    return log_folder


def save_config(config, path):
    """
    Saves a config as a json and pandas dataframe.

    Args:
        config (Config): Config.
        path (str): Path to save at.
    """
    dic = config.__dict__.copy()
    del (dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"])

    with open(path + ".json", "w") as f:
        json.dump(dic, f)


def upload_to_kaggle(folders, directory, dataset_name):
    """
    Uploads directories to a Kaggle dataset.

    Args:
        folders (list of strs): Folders to upload.
        directory (str): Path to save the dataset to.
        dataset_name (str): Name of the dataset.
    """
    os.mkdir(directory)

    for folder in folders:
        print(f"- Copying {folder}...")
        name = "_".join(folder[:-1].split('/')[-2:])
        shutil.copytree(folder, directory + name)

    # Create dataset-metadata.json
    with open(directory + 'dataset-metadata.json', "w") as f:
        slug = re.sub(' ', '-', dataset_name.lower())
        dic = {
            "title": f"{dataset_name}",
            "id": f"theoviel/{slug}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        json.dump(dic, f)

    # Upload dataset
    print('- Uploading ...')
    command = f"kaggle d create -p {directory} --dir-mode zip"

    try:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print('Output :', output)
        print('Error :', error)
    except Exception:
        print('Upload failed, Run command manually :', command)
