import numpy as np


def balance_ds(dataset, labels, seed=0):
    """Balance dataset

    Args:
        dataset: size nepoch, nchannel, nsamples
        labels: 1-D array of labels.

    Returns:
        tuple(np.array, np.array):
            dataset: np.array of balanced dataset.
            labels: 1-D np.array
    """
    np.random.seed(seed)
    # count the labels
    lvls, cnts = np.unique(labels, return_counts=True)
    mcnt = np.min(cnts)
    idxs = [np.random.choice(np.where(np.array(labels) == lvl)[0], size=mcnt) for lvl in lvls]
    return np.array(dataset)[np.concatenate(idxs)], np.array(labels)[np.concatenate(idxs)]