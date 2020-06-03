import pyeeg
import numpy as np
import pandas as pd

locate = lambda listOfElems, elem: [ i for i in range(len(listOfElems)) if listOfElems[i] == elem ]


def bin_power(dataset, fsamp, Band=range(0,45)):
    dataset_power = []
    for i, data in enumerate(dataset):
        res = []
        for j, channel in enumerate(data):
            power = pyeeg.bin_power(channel, Band=Band, Fs=fsamp)[0]
            res.append(power)
        dataset_power.append(res)
    return dataset_power


def bin_power_avg(dataset, fsamp, Band=range(0,45)):
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
