import pyeeg

from src.features.feature import power_and_freq, power
import numpy as np
import pandas as pd
import catch22

FEATURES = [
    'DN_HistogramMode_5',
    'DN_HistogramMode_10',
    'CO_f1ecac',
    'CO_FirstMin_ac',
    'CO_HistogramAMI_even_2_5',
    'CO_trev_1_num',
    'MD_hrv_classic_pnn40',
    'SB_BinaryStats_mean_longstretch1',
    'SB_TransitionMatrix_3ac_sumdiagcov',
    'PD_PeriodicityWang_th0_01',
    'CO_Embed2_Dist_tau_d_expfit_meandiff',
    'IN_AutoMutualInfoStats_40_gaussian_fmmi',
    'FC_LocalSimple_mean1_tauresrat',
    'DN_OutlierInclude_p_001_mdrmd',
    'DN_OutlierInclude_n_001_mdrmd',
    'SP_Summaries_welch_rect_area_5_1',
    'SB_BinaryStats_diff_longstretch0',
    'SB_MotifThree_quantile_hh',
    'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
    'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
    'SP_Summaries_welch_rect_centroid',
    'FC_LocalSimple_mean3_stderr',
    'PW_average_power',
    'PW_average_freq'
]


def get_features(dataset):
    """Dataset to features

    Args:
        dataset: Pre-processed dataset, of size (nepoch, nchannel, nsamples)

    Returns:
        tuple: size of nfeatures x nepoch.
            If there are channel dependent features, list them in ch1.a, ch1.b, ..., chn.a, chn.b
    """
    #TODO: configs as argumnets
    fsamp = 256
    band = range(0, 45)
    c22feas = [[feature_channel(channel, fsamp=fsamp, band=band) for channel in epoch] for epoch in dataset]
    c22feas = ds3to2(c22feas)
    return pd.DataFrame({'f'+str(i):feature for i, feature in enumerate(c22feas)})


def ds3to2(dataset):
    """Convert features dataset to pd.DataFrame

    Args:
        dataset: 3D list of nepoch x nchannel x nfeatures
    Returns:
        2D-array of row = nfeatures x nchannel;
                    col = nepoch
    """
    n1, n2, n3 = np.shape(dataset)
    return np.array(dataset).transpose([1,2,0]).reshape([n2*n3,n1])


# TODO: channel-wise feature
def feature_channel(channel, fsamp=256, band=range(0, 45)):
    """Convert time series of a given channel to list of features.
    Args:
        channel: 1-D array-like
        fsamp: sampling rate in Hz.
        band: band-pass in Hz.
    Returns:
        list: 1-D array-like
    """

    # catch 22
    res = catch22.catch22_all(channel)['values']

    # power and freq
    power = pyeeg.bin_power(channel, Band=band, Fs=fsamp)[0]
    pwd = np.mean(power)
    freqs = np.arange(0, len(power))
    pdf = np.array(power) / np.sum(power)
    mu = np.sum(freqs * pdf)
    # m2 = np.sum((freqs - mu)**2 * pdf)
    res.extend([pwd, mu])

    return res


def feature_2D(d2):
    """Convert a collection of time series to a list of features
    Args:
        d2: 2-D array-like, with shape of (nchannel x nsamples)
    Returns:
        list: 1-D array-like features.
    """
    raise NotImplementedError


# TODO: Move to utils
def dataset_to_df(dataset, labels):
    """Convert numpy array DATASET, LABELS to pd.DataFrame

    Args:
        dataset: size nepoch x nchannel x nsamples/nfeatures
        labels: nepoch x 1
    Returns: pd.DataFrame
        nrow = nepoch x nsamples;
        ncol = nchannel + 1
    """
    ds_pwd = np.array(dataset)
    ne, nc, ns = np.shape(ds_pwd)
    df_pwd = pd.DataFrame(ds_pwd.transpose([0,2,1]).reshape(-1, np.shape(ds_pwd)[1]),
                         columns=['ch'+ str(i) for i in np.arange(0, nc)])
    df_pwd = df_pwd.assign(labels = np.repeat(labels, ns))\
        .assign(epoch = np.repeat(np.arange(0, ne), ns))
    return df_pwd


# TODO: move to train_model
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
