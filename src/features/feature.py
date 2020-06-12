"""Feature Engineering-related function"""
import numpy as np
import pyeeg


def power_and_freq(data):
    """Calculate 1st moment of y and 1st, 2nd moment of x

    Args:
        data: num_of_channel x num_of_freq

    Returns:
        data: num_of_channel x 3
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
