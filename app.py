import streamlit as st
import numpy as np
from dataset_funcs import bin_power_avg
from sklearn import preprocessing
import pickle


def plot_eeg(data, fsamp=1):
    """Make a plot of eeg data

    Args:
        data: nchannel x nsample
        fsamp: samping rate. must be integer

    Returns:
        plot

    """


st.title('SeizureVista')

# Sampling rate
fsamp = st.text_input('Sampling rate (Hz)', value='400')
try:
    fsamp = int(fsamp)
    if fsamp <= 0:
        raise Exception
except:
    st.write('Samping rate must be positive integer')

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["txt", "csv"])

if uploaded_file is not None:
    ds = np.loadtxt(uploaded_file, delimiter=',')
    X = bin_power_avg([ds], fsamp=fsamp)

    with open('model.pkl', 'rb') as fp:
        clf = pickle.load(fp)

    res = clf.predict(X)[0]

    st.write(res)
