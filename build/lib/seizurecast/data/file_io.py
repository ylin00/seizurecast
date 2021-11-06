"""
Handling EDF file reading

Functions in this module should:

Args:
    path(str): string path to file or folder

Returns:
    dataset: nchannel x nsamples
"""
import glob
import os
import re

import pandas as pd

from seizurecast.data.tu_pystream import nedc_pystream as ps


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


def load_tse_bi(token_path):
    """Reads .tse_bi file and returns backgroud and seizure intervals

    Args:
        token_path: path to token file
            e.g. tuh_eeg/v1.0.0/edf/00000037/00001234/s001_2012_03_06
            /00001234_s001_t000

    Returns:
        tuple: intvs - list of intervals.   labels - list of labels.

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
