import psycopg2
import sys, os
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import notebooks.example_psql as creds
from sqlalchemy import create_engine
from src.data import file_io
from src.data.file_io import read_1_token, sort_channel, load_tse_bi, relabel_tse_bi, signal_to_dataset
from src.data.preprocess import preprocess
from src.features.dataset_funcs import get_features

# Create connection to postgresql
from src.models.par import STD_CHANNEL_01_AR

engine = create_engine(f'postgresql://{creds.PGUSER}:{creds.PGPASSWORD}@{creds.PGHOST}:5432/{creds.PGDATABASE}')


def new_tables():
    # directory
    df = file_io.listdir_edfs('/Users/yanxlin/github/ids/tusz_1_5_2/edf/')
    df = df.rename(columns={'path7': 'train_test'})
    df.to_sql('directory', con=engine, if_exists='replace')

    # seiz-bckg
    df = pd.read_table('/Users/yanxlin/github/ids/tusz_1_5_2/_DOCS/ref_train.txt', header=None, sep=' ',
                       names=['token', 'time_start', 'time_end', 'label', 'prob']).assign(train_test='train')
    df2 = pd.read_table('/Users/yanxlin/github/ids/tusz_1_5_2/_DOCS/ref_dev.txt', header=None, sep=' ',
                        names=['token', 'time_start', 'time_end', 'label', 'prob']).assign(train_test='test')
    df.append(df2).to_sql('seiz_bckg', engine, if_exists='replace')


def dataset_from_many_edfs(token_files, montage=STD_CHANNEL_01_AR, len_pre=100, len_post=300, sec_gap=0, fsamp=256):
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
