from future.future import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mne
import os


if __name__ == '__main__':
    """load one token"""
    train_path = '../tusz_1_5_2/edf/train'
    tcp_type = '01_tcp_ar'
    patient_group = '004'
    patient = '00000492'
    session = 's003_2003_07_18'
    token = '00000492_s003_t001'
    token_path = os.path.join(train_path, tcp_type, patient_group, patient,
                              session, token)

    # fsamp_mont, sig_mont, labels_mont = read_1_token(token_path)
    # print(labels_mont)
    # print(pd.DataFrame(sig_mont))

    print(load_tse_bi(token_path))



