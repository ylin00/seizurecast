import numpy as np
from seizurecast.data.file_io import read_1_token, load_tse_bi
from seizurecast.data.label import relabel_tse_bi
from seizurecast.data.preprocess import preprocess, sort_channel, signal_to_dataset
from seizurecast.models.parameters import STD_CHANNEL_01_AR


def make_dataset(token_files, montage=STD_CHANNEL_01_AR, len_pre=100, len_post=300, sec_gap=0, fsamp=256):
    """Read and process a list of edf files

    Args:
        token_files(list): list of edf file path without extension
        len_pre: length of pre-seizure stage in seconds
        len_post: length of post-seizure stage in seconds
        sec_gap: gap between pre-seizure and seizure in seconds
        fsamp(int): Desired sampling rate in Hz

    Returns:
        tuple(tuple, tuple, int): dataset, labels, sampling rate
    """
    # TODO: assert no extension in the edf file path
    # TODO: dataset return as tuples

    dataset, labels = [], []
    for tf in token_files:
        # load token
        f, s, l = read_1_token(tf)
        f = int(np.mean(f))

        # sort channel label
        s = sort_channel(s, l, std_labels=montage)

        # load labeling file
        intvs, labls = load_tse_bi(tf)

        # relabel
        intvs, labls = relabel_tse_bi(intvs=intvs, labels=labls,
                                      len_pre=len_pre, len_post=len_post,
                                      sec_gap=sec_gap)

        # pre process
        s = preprocess(s, fsamp / np.mean(f))

        # generate dataset
        ds, lbl = signal_to_dataset(raw=s, fsamp=fsamp, intvs=intvs,
                                    labels=labls)
        dataset.extend(ds)
        labels.extend(lbl)

    return dataset, labels
