from time import time
import cProfile, pstats, io
from file_io import get_all_edfs
from pipeline import Config, Pipeline


def run_code():
    edfs = get_all_edfs()
    conf = Config()
    pipe = Pipeline(conf)
    pipe.token_paths = edfs.sample(4,random_state=0)['token_path'].to_numpy()
    dump0 = time()
    print(f'Dumping {len(pipe.token_paths)} files')
    pipe.dump_xy()
    print(f'Dump cost = {round(time()-dump0)} s')
    pass

pr = cProfile.Profile()
pr.enable()
# ... do something ...

run_code()

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
sortby = 'tottime'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
ps.dump_stats('tmp/cprofile_dump')
print(s.getvalue())