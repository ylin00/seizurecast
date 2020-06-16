"""Feature Engineering-related function

Functions in this module must have the following syntax.

TODO: implement function interface following https://realpython.com/python-interface/
Args:
    data(list): (num_of_channel, num_of_sample)
Returns:
    list: (a list of features)  #TODO: change current interface to adhere to this

TODO: make use of catch 22
"""
import catch22
import numpy as np
import pandas as pd
import pyeeg

from seizurecast.utils import dataset_3d_to_2d


def power_and_freq(data):
    """Calculate 1st moment of y and 1st, 2nd moment of x

    Args:
        data: num_of_channel x num_of_freq

    Returns:
        data: num_of_channel x 2 (pwd, freq)
    """
    res0 = []
    for channel in data:
        pwd = np.mean(channel)
        freqs = np.arange(0, len(channel))
        pdf = np.array(channel) / np.sum(channel)
        mu = np.sum(freqs * pdf)
        # m2 = np.sum((freqs - mu)**2 * pdf)
        res0.append((pwd, mu))
    return res0


def power(data, fsamp:int, band=range(0, 45)):
    """Power spec of dataset

    Args:
        data: num_of_channel x num_of_sample

    """
    res = []
    for j, channel in enumerate(data):
        power = pyeeg.bin_power(channel, Band=band, Fs=fsamp)[0]
        res.append(power)
    return res


def line_length(data):
    """Line length

    References: https://link.springer.com/article/10.1007/s11517-019-02039-1

    Args:
        data: num_of_channel x num_of_sample

    Returns:
        list: num_of_channel x 1

    """
    return [np.mean(np.abs(np.diff(channel))) for channel in data]


def line_length_2(data, window_size=5):
    """Window average of line lengths.

    Args:
        data: num_of_channel x num_of_sample
        window_size:

    Returns:
        same length as line_lengths, taped with mean

    """
    res = []
    for channel in data:
        ll = np.abs(np.diff(channel))
        l2 = [np.mean(ll[max(0, i-window_size+1):i+1]) for i, _ in enumerate(
            ll)]
        res.append(np.mean(l2))
    return res


def line_length_3(data, ws1=5):
    res = []
    for channel in data:
        ll = np.abs(np.diff(channel))
        l2 = [np.mean(ll[max(0, i-ws1+1):i+1]) for i, _ in enumerate(
            ll)]
        l3 = np.abs(np.diff(l2))
        res.append(np.mean(l3))
    return res


def freq_of_burst(data):
    """Frequency of burst

    References:
        Litt, Brian, Rosana Esteller, Javier Echauz, Maryann D’Alessandro,
        Rachel Shor, Thomas Henry, Page Pennell, et al. 2001. “Epileptic
        Seizures May Begin Hours in Advance of Clinical Onset: A Report of
        Five Patients.” Neuron 30 (1): 51–64.
        https://doi.org/10.1016/S0896-6273(01)00262-8.

        Bursts were defined as sustained elevations of signal energy of
        greater than two standard deviations above the interburst baseline
        for periods of 5 min or more in a running average of 5 min duration.

    Args:
        data: num_of_sample x num_of_channel

    Returns:

    """
    pass


def RMSE(data):
    """"""
    return np.mean(np.array(data)**2, 0)
    pass


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


def get_features(dataset):
    """Dataset to features

    Args:
        dataset: Pre-processed dataset, of size (nepoch, nchannel, nsamples)

    Returns:
        pd.DataFrame: size of nfeatures x nepoch.
            If there are channel dependent features, list them in ch1.a, ch1.b, ..., chn.a, chn.b
    """
    #TODO: configs as argumnets
    fsamp = 256
    band = range(0, 45)
    c22feas = [[feature_channel(channel, fsamp=fsamp, band=band) for channel in epoch] for epoch in dataset]
    c22feas = dataset_3d_to_2d(c22feas)  # TODO: merge with dataset2Xy
    df = pd.DataFrame({'f'+str(i):feature for i, feature in enumerate(c22feas)})
    return df


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