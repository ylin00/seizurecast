from scipy import signal
import numpy as np
import pandas as pd

# TODO: rename s
def preprocess(s, resample_factor=1.0, freq_range=[0.01, 0.1]):
    """Pre process

    Args:
        s: signal. (nchannel x nsamples)
        resample_factor: resampling factor

    Returns:
        np.array: (nchannel x nsamples)
    """
    # Resampling
    if abs(resample_factor - 1) > 0.01:
        s = signal.resample(s,
                            num=int(np.rint(np.shape(s)[1] * resample_factor)),
                            axis=1)

    # read token and convert to data frame
    df0 = pd.DataFrame(np.array(s).transpose(), columns=['ch' + str(i) for i in range(0, len(s))])  # TODO: use l as label

    # Drop Channels
    df0 = df0.iloc[:, 0:8]

    # Remove DC offset
    df0 = df0.apply(lambda x: x - np.mean(x))

    # Filter with low and high pass
    filter = signal.firwin(400, freq_range, pass_zero=False)
    df0 = df0.apply(lambda x: np.real(signal.convolve(x.to_numpy(), filter, mode='same')))

    return df0.to_numpy().transpose()
