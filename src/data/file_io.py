"""
1. read edf
    . read edf returns raw: a list of lists
    . read .tse_bi, returns label_array: time array with labels.
    . take label_array and raw, returns dataset and label_array : a list of
        data, each block is 1 second, with fixed size. width is number of
        channels in certain standard order.
2. get labels
3. train models
4. validate models

Functions in this module should:

Args:
    path(str): string path to file or folder

Returns:
    dataset: nepoch x nchannel x nsamples
"""
import glob
import math
import os
import re

import pandas as pd

from src.data.tu_pystream import nedc_pystream as ps
from src.models.par import LABEL_BKG, LABEL_PRE, LABEL_SEZ, LABEL_POS, LABEL_NAN, \
    STD_CHANNEL_01_AR


def listdir_edfs(directory='~/github/ids/tusz_1_5_2/edf/train/01_tcp_ar',
                 columns=('tcp_type', 'patient_group', 'patient',
                          'session', 'token')):
    """Returns all edf filepaths in a DataFrame

    Returns:
        pd.DataFrame: filepaths
    """
    filelist = glob.glob(os.path.join(directory, '**', '*.edf'), recursive=True)
    fparts = [re.split('/|[.]edf', filename)[:-1] for filename in filelist]

    if len(fparts[0]) > len(columns):
        columns = ['path'+str(i) for i in range(0, len(fparts[0])-len(columns))] + list(columns)

    df = pd.DataFrame({key: value for key, value in zip(tuple(columns), tuple(zip(*fparts)))})

    # A very complicated lambda function
    return df.assign(token_path=lambda x: eval(
        """eval("+'/'+".join(["x.""" + '","x.'.join(x.columns) + '"]))'))


def read_1_token(token_path):
    """Read EDF file and apply its corresponding montage info

    Args:
        token_path: file path to the edf file. Should follow the same
            nameing rules defined in
            https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg/_DOCS/conventions/filenames_v00.txt

    Returns:
        tuple: 1-list of sampling rate. 2-list of list of signals in micro
            Volt. 3-list of labels.

    """

    """Load parameters"""
    params = ps.nedc_load_parameters_lbl(token_path + '.lbl')

    """read edf"""
    fsamp, sig, labels = ps.nedc_load_edf(token_path + '.edf')

    """select channels"""
    fsamp_sel, sig_sel, labels_sel = ps.nedc_select_channels(params, fsamp, sig,
                                                             labels)

    """apply montage"""
    fsamp_mont, sig_mont, labels_mont = ps.nedc_apply_montage(params, fsamp_sel,
                                                              sig_sel,
                                                              labels_sel)
    return fsamp_mont, sig_mont, labels_mont


def read_1_session(session_path):
    """Read multiple tokens of the same session

    Args:
        session_path: file path to a session.
            e.g. tuh_eeg/v1.0.0/edf/00000037/00001234/s001_2012_03_06

    Returns:
        dataset, labels
    """
    raise NotImplementedError


def read_1_patient(patient_folder):
    """Read multiple sessions belonging to the same patient

    Args:
        patient_folder: path to a patient folder
            e.g.: tuh_eeg/v1.0.0/edf/00000037

    Returns:
        standard data TBD

    """
    pass


def load_tse_bi(token_path):
    """Reads .tse_bi file and returns backgroud and seizure intervals

    Args:
        token_path: path to token file
            e.g. tuh_eeg/v1.0.0/edf/00000037/00001234/s001_2012_03_06
            /00001234_s001_t000

    Returns:
        intvs: list of intervals
        labels: list of labels

    """
    intvs, labels = [], []
    with open(token_path+'.tse_bi', 'r') as fp:
        for line in fp:
            line = line.replace('\n', '').split(' ')
            if line[0] == 'version':
                if line[2] != 'tse_v1.0.0':
                    print("tse_bi file must be version='tse_v1.0.0'")
                    exit(-1)
                else:
                    continue
            elif line[0] is not '':
                intvs.append([float(line[0]), float(line[1])])
                labels.append(line[2])
            else:
                continue
    return intvs, labels


def relabel_tse_bi(intvs, labels, len_pre=100, len_post=300, sec_gap=0):
    """ Compute labels from background and seizure intervals

    Args:
        intvs: list of list of intervals
        labels: list of labels of background and seizure. Must be same len as INTVS
        len_pre: length of pre-seizure stage in seconds
        len_post: length of post-seizure stage in seconds
        sec_gap: gap between pre-seizure and seizure in seconds

    Returns:
        tuple: (Array of intervals, Array of labels).
    """
    _intvs, _labls = [], []
    # Find LABEL_SEZ
    # locate pre, gap and pos
    # assign others to LABEL_BKG
    for i, lbl in enumerate(labels):
        beg, end = intvs[i]
        pos = beg + len_post
        pre = end - sec_gap - len_pre
        gap = end - sec_gap
        if lbl == LABEL_BKG:
            if i == 0 and i == len(labels)-1:
                _intvs.append([beg, end])
                _labls.append(LABEL_BKG)
            elif i == 0:
                _intvs.extend([[beg, max(beg, pre)],
                               [max(beg, pre), max(beg, gap)],
                               [max(beg, gap), end]])
                _labls.extend([LABEL_BKG, LABEL_PRE, LABEL_NAN])
            elif i == len(labels)-1:
                _intvs.extend([[beg, min(pos, end)],
                               [min(pos, end), end]])
                _labls.extend([LABEL_POS, LABEL_BKG])
            else:
                pos_end = min(pos, end)
                pre_beg = max(pos_end, pre)
                gap_beg = max(pos_end, gap)
                _intvs.extend([[beg, pos_end],[pos_end, pre_beg],
                               [pre_beg, gap_beg],[gap_beg, end]])
                _labls.extend([LABEL_POS, LABEL_BKG, LABEL_PRE, LABEL_NAN])
        elif lbl == LABEL_SEZ:
            _intvs.append([beg, end])
            _labls.append(LABEL_SEZ)
        else:
            raise ValueError("Illegal LABELS")

    _labls = [_labls[i] for i, intv in enumerate(_intvs) if intv[0] < intv[1]]
    _intvs = [_intvs[i] for i, intv in enumerate(_intvs) if intv[0] < intv[1]]

    return _intvs, _labls


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
