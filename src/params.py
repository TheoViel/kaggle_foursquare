import torch
import numpy as np

NUM_WORKERS = 4

DATA_PATH = "../data/"
IMG_PATH = DATA_PATH + "imgs/"
LOG_PATH = "../logs/"
OUT_PATH = "../output/"

CLASSES = ["aligned", "dirty", "misaligned"]

NUM_CLASSES = 3

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
