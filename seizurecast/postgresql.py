import pandas as pd
from sqlalchemy import create_engine

import seizurecast.config as creds
from seizurecast.data import file_io, label

# TODO: move this to setup/config.ini
# Create connection to postgresql
from seizurecast.data.make_dataset import make_dataset
from seizurecast.feature import get_features

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


def __feature_1_token(tk, fsamp=256, verbose=False, feature_type='c22'):
    """Generate feature from 1 token file"""
    print(f"Processing token: ...{tk[-14:]}") if verbose else None

    ds, _ = make_dataset([tk], len_pre=0, len_post=0, sec_gap=0, fsamp=fsamp)

    df = get_features(ds, feature_type=feature_type)

    intvs, lbls = file_io.load_tse_bi(tk)
    upperbounds = tuple(zip(*intvs))[1]

    df = df.assign(post=lambda df: label.post_sezure_s(df.index + 1, upperbounds, lbls),
                     pres=lambda df: label.pres_seizure_s(df.index + 1, upperbounds, lbls))

    return df


def write_features_to_sql_(
        indexes=(0, -1),
        verbose=True,
        query="select token, token_path from directory where train_test = 'dev' and tcp_type = '01_tcp_ar';",
        target_table='feature192_dev_01',
        feature_type='c22'):
    """
    Read edf paths from directory table, convert to features and write to given table.

    Args:
        indexes(tuple): (start, end) the range of index to write to sql.
        verbose:

    Returns:

    """
    fsamp = 256
    beg, end = indexes

    print(query, feature_type, target_table) if verbose else None
    tks = pd.read_sql(query, SQLengine)

    nbatch = tks.shape[0]
    for (index, Series) in tks.iloc[beg:end, :].iterrows():

        print(f"Processing batch {str(index)}/{str(nbatch)}")
        df = __feature_1_token(Series['token_path'], fsamp=fsamp, verbose=verbose, feature_type=feature_type) \
            .assign(token=Series['token'])

        df.to_sql(target_table, SQLengine, if_exists='append')

        del df


def write_features_to_sql(indexes=(0, -1), task='test-c22'):
    if task == 'test-c22':
        write_features_to_sql_(
            indexes=indexes, verbose=True,
            query="select token, token_path from directory where train_test = 'dev' and tcp_type = '01_tcp_ar';",
            target_table='feature192_dev_01',
            feature_type='c22'
        )
    elif task == 'train-256hz':
        write_features_to_sql_(
            indexes=indexes, verbose=True,
            query="select token, token_path from directory where train_test = 'train' and tcp_type = '01_tcp_ar';",
            target_table='train256hz_01',
            feature_type='hz256'
        )
    elif task == 'test-256hz':
        write_features_to_sql_(
            indexes=indexes, verbose=True,
            query="select token, token_path from directory where train_test = 'dev' and tcp_type = '01_tcp_ar';",
            target_table='test256hz_01',
            feature_type='hz256'
        )
    else:
        raise NotImplementedError


if __name__ == '__main__':

    write_features_to_sql()
    print(pd.read_sql_table('features', SQLengine).shape)
