import torch

OUT_PATH = "../output/"
DATA_PATH = "../data/"
RESSOURCES_PATH = "../data/ressources/"

IS_TEST = False
DEBUG = False

LOG_PATH = "../logs/"
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
