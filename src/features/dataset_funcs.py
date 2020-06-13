import pyeeg

from src.features.feature import power_and_freq, power
import numpy as np
import pandas as pd


def dataset_to_df(dataset, labels):
    """Convert numpy array DATASET, LABELS to pd.DataFrame"""
    ds_pwd = np.array(dataset)
    ne, nc, ns = np.shape(ds_pwd)
    df_pwd = pd.DataFrame(ds_pwd.transpose([0,2,1]).reshape(-1, np.shape(ds_pwd)[1]),
                         columns=['ch'+ str(i) for i in np.arange(0, nc)])
    df_pwd = df_pwd.assign(labels = np.repeat(labels, ns))\
        .assign(epoch = np.repeat(np.arange(0, ne), ns))
    return df_pwd


def balance_ds(dataset, labels, seed=0):
    """Balance dataset"""
    np.random.seed(seed)
    # count the labels
    lvls, cnts = np.unique(labels, return_counts=True)
    mcnt = np.min(cnts)
    idxs = [np.random.choice(np.where(np.array(labels) == lvl)[0], size=mcnt) for lvl in lvls]
    return np.array(dataset)[np.concatenate(idxs)], np.array(labels)[np.concatenate(idxs)]


def get_power_freq(dataset_power):
    """

    Returns:
        pd.DataFrame: with channels, powers and frequency
    """
    ds_pwd = []
    for epoch in dataset_power:
        res0 = power_and_freq(epoch)
        ds_pwd.append(res0)
    return ds_pwd


def get_power(dataset, fsamp:int, band=range(0, 45)):
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


# TODO: bin_power_avg == get_power ? change name
def bin_power_avg(dataset, fsamp, Band=range(0,45)):
    """"""
    DeprecationWarning()
    dataset_power = bin_power(dataset, fsamp, Band=Band)

    ds_pwd = np.array(dataset_power)
    # wrangling to dataframe
    ne, nc, ns = np.shape(ds_pwd)
    df_pwd = pd.DataFrame(
        ds_pwd.transpose([0, 2, 1]).reshape(-1, np.shape(ds_pwd)[1]),
        columns=['ch' + str(i) for i in np.arange(0, nc)])

    df_pwd = df_pwd.assign(
        freq=np.tile(np.arange(0, np.shape(ds_pwd)[2]), np.shape(ds_pwd)[0])) \
        .assign(epoch=np.repeat(np.arange(0, ne), ns))

    dfpwd = pd.wide_to_long(df_pwd, ['ch'], ['freq', 'epoch'],
                            'channel')
    res = []
    for name, group in dfpwd.groupby(['epoch', 'channel']):
        ng = group.reset_index() \
            .assign(pwd=lambda x: np.mean(x.ch),
                    freqs=lambda x: np.sum(x.freq * x.ch / np.sum(x.ch))) \
            .drop(['ch', 'freq'], 'columns') \
            .drop_duplicates(['epoch', 'channel'])
        res.append(ng)
    dfpwd2 = pd.concat(res)
    Xy = dfpwd2.pivot(index='epoch', columns='channel',
                      values=['pwd', 'freqs']).to_numpy()

    return Xy


# TODO: bin_power == get_power? remove reducdent
def bin_power(dataset, fsamp, Band=range(0,45)):
    dataset_power = []
    for i, data in enumerate(dataset):
        res = []
        for j, channel in enumerate(data):
            power = pyeeg.bin_power(channel, Band=Band, Fs=fsamp)[0]
            res.append(power)
        dataset_power.append(res)
    return dataset_power


locate = lambda listOfElems, elem: [ i for i in range(len(listOfElems)) if listOfElems[i] == elem ]
