from src.features.feature import power_and_freq, power
import numpy as np


# TODO: move to train_model
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


def bin_power_freq(dataset_power):
    """
    Args:
        dataset_power: list of size n_epoch x n_channel x n_freq
    Returns:
        list: nepoch x n channel x 2
    """
    ds_pwd = []
    for epoch in dataset_power:
        res0 = power_and_freq(epoch)
        ds_pwd.append(res0)
    return ds_pwd


def bin_power(dataset, fsamp:int, band=range(0, 45)):
    """Power spec

    Args:
        dataset: n_epoch x n_channel x n_sample
        fsamp:
        band:

    Returns:
        n_epoch x n_channel x len(band)
    """
    res = []
    for i, data in enumerate(dataset):
        res.append(power(data, fsamp=fsamp, band=band))
    return res
