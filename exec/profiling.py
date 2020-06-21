"""
This script profiles the time cost of data preprocessing.

"""
from time import time
import cProfile, pstats, io
from seizurecast.data.file_io import listdir_edfs
from seizurecast.features.to_sql import write_features_to_sql
from seizurecast.models.pipeline import Config, Pipeline


# ### functions to profile
def __pipeline():
    edfs = listdir_edfs()
    conf = Config()
    pipe = Pipeline(conf)
    pipe.token_paths = edfs.sample(4,random_state=0)['token_path'].to_numpy()
    dump0 = time()
    print(f'Dumping {len(pipe.token_paths)} files')
    pipe.dump_xy()
    print(f'Dump cost = {round(time()-dump0)} s')
    pass


def __write_features_to_sql():
    write_features_to_sql()


pr = cProfile.Profile()
pr.enable()

# ### do something ###

__write_features_to_sql()

# ######### end ########## #

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
sortby = 'tottime'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
ps.dump_stats('../tmp/cprofile_dump')
print(s.getvalue())