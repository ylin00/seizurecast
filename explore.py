from io import *
import utils
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

# relabeling config
LEN_PRE = 15
LEN_POS = 60
SEC_GAP = 0


def load_a_session():

    """load one session"""
    train_path = '../tusz_1_5_2/edf/train'
    tcp_type = '01_tcp_ar'
    patient_group = '004'
    patient = '00000492'
    session = 's003_2003_07_18'
    token = '00000492_s003_t001'

    session_path = os.path.join(train_path, tcp_type, patient_group, patient,
                                session)

    token_files = [os.path.join(session_path, f[:-4]) for f in os.listdir(
        session_path) if f.endswith('.edf')]

    dataset, labels = [], []
    for tf in token_files:
        f, s, l = read_1_token(tf)
        # sort channel label
        s = sort_channel(s, l, STD_CHANNEL_01_AR)
        f = int(np.mean(f))
        intvs, labls = load_tse_bi(tf)
        intvs, labls = relabel_tse_bi(intvs=intvs, labels=labls,
                                      len_pre=LEN_PRE, len_post=LEN_POS,
                                      sec_gap=SEC_GAP)
        ds, lbl = signal_to_dataset(sig=s, fsamp=f, intvs=intvs, labels=labls)
        dataset.extend(ds)
        labels.extend(lbl)

    utils.save(dataset, labels)
    return dataset, labels


# load_a_session()
dataset, labels = utils.load()
print(np.shape(dataset), np.shape(labels))
