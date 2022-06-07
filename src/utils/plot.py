import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from sklearn.metrics import confusion_matrix


def plot_importances(imps):
    imps.index.name = "index"
    importances = imps.rename(columns={0: 'importance', "0": 'importance'})

    importances['importance'] = importances['importance'].apply(np.abs)
    importances = importances.sort_values("importance", ascending=False).reset_index()
    importances = importances[importances['importance'] != 0]

    plt.figure(figsize=(15, 25))
    sns.barplot(x='importance', y="index", data=importances)
    plt.yticks(fontsize=11)
    plt.ylabel(None)


def plot_confusion_matrix(
    y_pred,
    y_true,
    normalize=None,
    display_labels=None,
    cmap="viridis",
):
    """
    Computes and plots a confusion matrix.

    Args:
        y_pred (numpy array): Predictions.
        y_true (numpy array): Truths.
        normalize (bool or None, optional): Whether to normalize the matrix. Defaults to None.
        display_labels (list of strings or None, optional): Axis labels. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "viridis".
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    cm = cm[::-1, :]

    # Display colormap
    n_classes = cm.shape[0]
    im_ = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    # Display values
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text = f"{cm[i, j]:.0f}" if normalize is None else f"{cm[i, j]:.3f}"
        plt.text(
            j, i, text, ha="center", va="center", color=color
        )

    # Display legend
    plt.xlim(-0.5, n_classes - 0.5)
    plt.ylim(-0.5, n_classes - 0.5)
    plt.xticks(np.arange(n_classes), display_labels)
    plt.yticks(np.arange(n_classes), display_labels[::-1])

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
