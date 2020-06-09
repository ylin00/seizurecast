import pandas as pd
import glob
import os


def get_all_edfs():
    """Returns all edf filepaths in a DataFrame
    
    Returns:
        pd.DataFrame: filepaths
    """
    columns=('path0','path1','path2','path3', 'tcp_type', 'patient_group', 'patient', 'session', 'token')

    filelist = glob.glob(os.path.join('../tusz_1_5_2/edf/train/01_tcp_ar', '**', '*.edf'), recursive=True)
    fparts = [filename.split('/') for filename in filelist]

    df = pd.DataFrame({key:value for key, value in zip(tuple(columns), tuple(zip(*fparts)))})

    # A very complicated lambda function
    return df.assign(token_path = lambda x: eval("""eval("+'/'+".join(["x."""+'","x.'.join(x.columns)+'"]))'))


