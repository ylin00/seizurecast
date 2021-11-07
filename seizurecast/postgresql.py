import pandas as pd
from sqlalchemy import create_engine

import seizurecast.config as creds
from seizurecast.data import file_io, label

# TODO: move this to setup/config.ini
# Create connection to postgresql
from seizurecast.data.make_dataset import make_dataset, produce_signal
from seizurecast.feature import get_features
from seizurecast.data.parameters import STD_CHANNEL_01_AR
from seizurecast.utils import psql_insert_copy


#TODO: check to makesure POSTGRESQL is available otherwise return mocking engine
SQLengine = create_engine(f'postgresql://{creds.PGUSER}:{creds.PGPASSWORD}@{creds.PGHOST}:5432/{creds.PGDATABASE}',
                            executemany_mode = "batch")


def setup_directory(homedir="/Users/yanxlin/github/ids/tusz_1_5_2/edf"):
    """ Prepare directory table

    Args:
        homedir: home directory to the edf files

    """
    # directory
    df = file_io.listdir_edfs(homedir)
    df = df.rename(columns={'path6': 'train_test'})
    df.to_sql('directory', con=SQLengine, if_exists='replace')

    # seiz-bckg
    # df = pd.read_table('/Users/yanxlin/github/ids/tusz_1_5_2/_DOCS/ref_train.txt', header=None, sep=' ',
    #                   names=['token', 'time_start', 'time_end', 'label', 'prob']).assign(train_test='train')
    # df2 = pd.read_table('/Users/yanxlin/github/ids/tusz_1_5_2/_DOCS/ref_dev.txt', header=None, sep=' ',
    #                    names=['token', 'time_start', 'time_end', 'label', 'prob']).assign(train_test='test')
    # df.append(df2).to_sql('seiz_bckg', SQLengine, if_exists='replace')


def run_sql_task(indexes=(0, -1), task='test-c22'):
    """

    Args:
        indexes: id of the files obtained from the query to process.
        task: task code to select specific task

    """
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
    elif task == 'preprocessed_train_tcp01':
        import_edf_to_sql(indexes=indexes)
    else:
        raise NotImplementedError


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


def import_edf_to_sql(
        indexes=(0, -1), verbose=True,
        query="select token, token_path from directory where train_test = 'train' and tcp_type = '01_tcp_ar';",
        target_table='preprocessed_train_tcp01',
        montage=STD_CHANNEL_01_AR,
        fsamp=256):
    """
    Args:
        indexes: id of the files obtained from the query to process.
        query: SQL query to generate token_path.
        target_table: the new table to insert the processed results into
        verbose: verbose mode

    """

    print("executing query \n"+query, "\nInserting into table "+target_table) if verbose else None

    tks = pd.read_sql(query, SQLengine)
    nbatch = tks.shape[0]
    beg, end = indexes

    dfs = []
    for (index, Series) in tks.iloc[beg:end, :].iterrows():

        print(f"Processing batch {str(index)}/{str(nbatch)}")

        tk = Series['token_path']

        token = Series['token']

        s = produce_signal(tk, montage=montage, fsamp=fsamp)

        ts = pd.Series(range(0, len(s[0]))) / fsamp  # Assign timestamps in second

        df = pd.DataFrame({'ch' + str(i): fea for i, fea in enumerate(s)})\
            .assign(token=token, timestamp=ts)

        # assign pre and post seizure duration
        intvs, lbls = file_io.load_tse_bi(tk)
        upperbounds = tuple(zip(*intvs))[1]

        df = df.assign(post=lambda df: label.post_sezure_s(ts.to_numpy(), upperbounds, lbls),
                       pres=lambda df: label.pres_seizure_s(ts.to_numpy(), upperbounds, lbls))

        dfs.append(df)

    pd.concat(dfs).to_sql(target_table, SQLengine, if_exists='append', method=psql_insert_copy)


if __name__ == '__main__':

    #run_sql_task()
    setup_directory("/media/ylin00/swap/tusz_1_5_2/edf")
    print(pd.read_sql_table('directory', SQLengine).shape)
