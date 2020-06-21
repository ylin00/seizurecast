import pandas as pd
from sqlalchemy import create_engine

import seizurecast.postgresql as creds
from seizurecast.data import file_io, label

# TODO: move this to setup/config.ini
# Create connection to postgresql
from seizurecast.data.make_dataset import make_dataset
from seizurecast.features.feature import get_features

SQLengine = create_engine(f'postgresql://{creds.PGUSER}:{creds.PGPASSWORD}@{creds.PGHOST}:5432/{creds.PGDATABASE}')


def write_tables_to_sql():
    # directory
    df = file_io.listdir_edfs('/Users/yanxlin/github/ids/tusz_1_5_2/edf/')
    df = df.rename(columns={'path7': 'train_test'})
    df.to_sql('directory', con=SQLengine, if_exists='replace')

    # seiz-bckg
    df = pd.read_table('/Users/yanxlin/github/ids/tusz_1_5_2/_DOCS/ref_train.txt', header=None, sep=' ',
                       names=['token', 'time_start', 'time_end', 'label', 'prob']).assign(train_test='train')
    df2 = pd.read_table('/Users/yanxlin/github/ids/tusz_1_5_2/_DOCS/ref_dev.txt', header=None, sep=' ',
                        names=['token', 'time_start', 'time_end', 'label', 'prob']).assign(train_test='test')
    df.append(df2).to_sql('seiz_bckg', SQLengine, if_exists='replace')


def __feature_1_token(tk, fsamp=256, verbose=False):
    """Generate feature from 1 token file"""
    print(f"Processing token: ...{tk[-14:]}") if verbose else None

    ds, _ = make_dataset([tk], len_pre=0, len_post=0, sec_gap=0, fsamp=fsamp)

    df = get_features(ds)

    intvs, lbls = file_io.load_tse_bi(tk)
    upperbounds = tuple(zip(*intvs))[1]

    df = df.assign(post=lambda df: label.post_sezure_s(df.index + 1, upperbounds, lbls),
                     pres=lambda df: label.pres_seizure_s(df.index + 1, upperbounds, lbls))

    return df


def write_features_to_sql(indexes=(0, -1), verbose=True):
    """
    Read edf paths from directory table, convert to features and write to given table.

    Args:
        indexes(tuple): (start, end) the range of index to write to sql.
        verbose:

    Returns:

    """
    fsamp = 256
    query = "select token, token_path from directory where train_test = 'dev' and tcp_type = '01_tcp_ar';"
    target_table = 'feature192_dev_01'
    beg, end = indexes

    print("Only touch the test set and the tcp_type of 01") if verbose else None
    tks = pd.read_sql(query, SQLengine)

    nbatch = tks.shape[0]
    for (index, Series) in tks.iloc[beg:end, :].iterrows():

        print(f"Processing batch {str(index)}/{str(nbatch)}")
        df = __feature_1_token(Series['token_path'], fsamp=fsamp, verbose=verbose) \
            .assign(token=Series['token'])

        df.to_sql(target_table, SQLengine, if_exists='append')

        del df


if __name__ == '__main__':

    write_features_to_sql()
    print(pd.read_sql_table('features', SQLengine).shape)
