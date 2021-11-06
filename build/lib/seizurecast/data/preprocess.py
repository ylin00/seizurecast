import math
from scipy import signal
import numpy as np
import pandas as pd

from seizurecast.models.parameters import STD_CHANNEL_01_AR


# TODO: rename s
# TODO: remove dependence on dataframe
def preprocess(s, resample_factor=1.0, freq_range=[0.01, 0.1]):
    """Pre process

    Args:
        s: signal. (nchannel x nsamples)
        resample_factor: resampling factor

    Returns:
        np.array: (nchannel x nsamples)
    """
    # Resampling
    if abs(resample_factor - 1) > 0.01:
        s = signal.resample(s,
                            num=int(np.rint(np.shape(s)[1] * resample_factor)),
                            axis=1)

    # read token and convert to data frame
    df0 = pd.DataFrame(np.array(s).transpose(), columns=['ch' + str(i) for i in range(0, len(s))])  # TODO: use l as label

    # Drop Channels
    df0 = df0.iloc[:, 0:8]  #TODO: drop channels use input argument instead of 0:8

    # Remove DC offset
    df0 = df0.apply(lambda x: x - np.mean(x))

    # Filter with low and high pass
    filter = signal.firwin(400, freq_range, pass_zero=False)
    df0 = df0.apply(lambda x: np.real(signal.convolve(x.to_numpy(), filter, mode='same')))

    return df0.to_numpy().transpose()


def sort_channel(raw, ch_labels, std_labels=STD_CHANNEL_01_AR):
    """sort channel based on standard labels

    Args:
        raw: (n_channel, n_sample) of EEG signals.
        ch_labels: array of channel labels. Len(LABELS) must = width of SIG
        std_labels: array of standard channel labels. must of same len as LABELS

    Returns:
        list: EEG signals, same shape as RAW

    """
    if len(set(ch_labels).intersection(set(std_labels))) < len(std_labels):
        raise Exception('Channel labels must match the length of std_labels')
    else:
        return [raw[i] for i in [ch_labels.index(lbl) for lbl in std_labels]]


def chop_signal(raw, n_sample_per_epoch:int):
    """Generate dataset from EEG signals and labels

    Args:
        raw: EEG signals. Shape: (n_channel, n_sample).
        n_sample_per_epoch: Number of samples per epoch.

    Returns:
        list: EEG signals (n_epochs, n_channels, n_sample_per_epoch).

    """
    n_times = len(raw[0])
    res = []
    for i in range(0, n_times // int(n_sample_per_epoch), 1):
        res.append([channel[i * n_sample_per_epoch:(i + 1) * n_sample_per_epoch] for channel in raw])
    return res


def signal_to_dataset(raw, fsamp, intvs, labels):
    """Segmentize raw data into list of epochs.

    returns dataset and label_array : a list of data, each block is 1
        second, with fixed size. width is number of channels in certain standard
        order.

    Args:
        raw: EEG signals. Shape: (n_channel, n_sample).
        fsamp(int): sampling rate, i.e., window size of resulting epoch. Unit: Hz
        intvs: list of [start, end]. Unit: second
        labels: list of labels. Must be same len as INTVS

    Returns: tuple (dataset, labels):
            - dataset: list of data; (n_epochs, n_channels, n_sample_per_epoch)
            - labels: list of labels

    """
    ds, lbl = [], []
    for i, inv in enumerate(intvs):
        tstart, tend = inv
        chopped_sig = chop_signal(
            [ch[math.ceil(tstart*fsamp):math.floor(tend*fsamp)] for ch in raw],
            fsamp)
        ds.extend(chopped_sig)
        lbl.extend([labels[i]] * len(chopped_sig))
    return ds, lbl
