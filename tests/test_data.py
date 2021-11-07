from seizurecast.data.file_io import *
from seizurecast.data.label import relabel_tse_bi
from seizurecast.data.preprocess import sort_channel, chop_signal, signal_to_dataset
from seizurecast.data.parameters import LABEL_BKG, LABEL_PRE, LABEL_SEZ, LABEL_POS
import numpy as np


def test_relabel():
    intvs = [[0, 400], [400, 500], [500, 800], [800, 900], [900, 1200]]
    labels = ([LABEL_BKG] + [LABEL_SEZ]) * 2 + [LABEL_BKG]
    intvs, labs = relabel_tse_bi(intvs, labels, len_pre=100, len_post=100,
                                 sec_gap=0)
    # 300*bkg + 100*pre + 100*sez + 100*post + 100*bkg + 100*pre + 100*sez +
    # 100*post
    np.testing.assert_array_equal(labs, [LABEL_BKG, LABEL_PRE, LABEL_SEZ,
                                         LABEL_POS] * 2 + [LABEL_BKG])


def test_relabel_long_pre():
    intvs = [[0, 400], [400, 500], [500, 800], [800, 900], [900, 1000]]
    labels = ([LABEL_BKG] + [LABEL_SEZ]) * 2 + [LABEL_BKG]
    intvs, labs = relabel_tse_bi(intvs, labels, len_pre=400, len_post=100,
                                 sec_gap=0)
    # 300*bkg + 100*pre + 100*sez + 100*post + 100*bkg + 100*pre + 100*sez +
    # 100*post
    np.testing.assert_array_equal(labs, [LABEL_PRE, LABEL_SEZ,
                                         LABEL_POS] * 2)


def test_relabel_long_post():
    intvs = [[0, 400], [400, 500], [500, 800], [800, 900], [900, 1000]]
    labels = ([LABEL_BKG] + [LABEL_SEZ]) * 2 + [LABEL_BKG]
    intvs, labs = relabel_tse_bi(intvs, labels, len_pre=100, len_post=250,
                                 sec_gap=0)
    # 300*bkg + 100*pre + 100*sez + 100*post + 100*bkg + 100*pre + 100*sez +
    # 100*post
    np.testing.assert_array_equal(
        labs,
        [LABEL_BKG, LABEL_PRE, LABEL_SEZ, LABEL_POS, LABEL_PRE, LABEL_SEZ,
         LABEL_POS])


def test_relabel_long_pre_post():
    intvs = [[0, 400], [400, 500], [500, 800], [800, 900], [900, 1000]]
    labels = ([LABEL_BKG] + [LABEL_SEZ]) * 2 + [LABEL_BKG]
    intvs, labs = relabel_tse_bi(intvs, labels, len_pre=100, len_post=300,
                                 sec_gap=0)
    # 300*bkg + 100*pre + 100*sez + 100*post + 100*bkg + 100*pre + 100*sez +
    # 100*post
    np.testing.assert_array_equal(
        labs,
        [LABEL_BKG, LABEL_PRE, LABEL_SEZ, LABEL_POS, LABEL_SEZ, LABEL_POS])

    # alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    #             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    # labels = alphabet[np.arange(0,12)]


def test_chop_signal_undersample():
    sig = [np.arange(0, 11)] * 12  # 12 x 11 matrix
    # function chop_signal should takes 2 arguments
    data = chop_signal(sig, 3)
    np.testing.assert_equal(len(data), 3)  # n_epoch
    np.testing.assert_equal(len(data[0]), 12)  # n_channel
    np.testing.assert_equal(len(data[0][0]), 3)  # n_times


def test_chop_signal():
    sig = [np.arange(0, 11)] * 12  # 11 x 12 matrix
    data = chop_signal(sig, 11)
    np.testing.assert_equal(len(data), 1)  # 1 x 11 x 12
    np.testing.assert_equal(len(data[0]), 12)
    np.testing.assert_equal(len(data[0][0]), 11)


def test_chop_supersample():
    sig = [np.arange(0, 11)] * 12  # 11 x 12 matrix
    data = chop_signal(sig, 12)
    np.testing.assert_equal(len(data), 0)  # 0


def test_load_tse_bi():
    path = './tests/data/train/tcp_type/patient_group/patient/session/token_t001'
    intvs, labls = load_tse_bi(path)
    np.testing.assert_array_equal(intvs, [[0.0, 33.1425], [33.1425, 53.0008],
                                          [53.0008, 184.0]])
    np.testing.assert_array_equal(labls, ['bckg', 'seiz', 'bckg'])


def test_signal_to_dataset():
    n_channels, n_times = 12, 368
    sig = [np.arange(0, n_times)] * n_channels  # 368 x 12 matrix
    intvs = [[0.0, 33.1425], [33.1425, 53.0008], [53.0008, 184.0]]
    labls = ['a', 'b', 'c']
    fsamp = 2

    data, labels = signal_to_dataset(sig, fsamp, intvs, labls)
    np.testing.assert_equal(len(data), 33 + 19 + 130)  # 1 epoch per second
    np.testing.assert_equal(len(data), len(labels))
    np.testing.assert_array_equal(labels[0:33], ['a'] * 33)
    np.testing.assert_array_equal(labels[33:33 + 19], ['b'] * 19)


def test_sort_channel():
    sig=[[1], [2], [3]]  # 3 channel
    chl=['a', 'b', 'c']  # channel label
    newsig = sort_channel(sig, chl, std_labels=['c', 'b', 'a'])
    np.testing.assert_array_equal(newsig, [[3], [2], [1]])


def test_sort_channel_drop_missing():
    sig=[[1], [2]]  # 3 channel
    chl=['a', 'b']  # channel label
    with np.testing.assert_raises(Exception):
        newsig = sort_channel(sig, chl, std_labels=['c', 'b', 'a'])
    # np.testing.assert_array_equal(newsig, None)


def test_sort_channel_extra_channel():
    sig=[[1], [2], [3], [4]]  # 4 channel
    chl=['a', 'b', 'c', 'd']  # channel label
    newsig = sort_channel(sig, chl, std_labels=['c', 'b', 'a'])
    np.testing.assert_array_equal(newsig, [[3], [2], [1]])
