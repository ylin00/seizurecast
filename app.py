import time

import streamlit as st
import numpy as np

from EEGStreamProcessor import EEGStreamProcessor, StreamerOptions
from EEGStreamer import sleep_and_sync, decode
from dataset_funcs import bin_power_avg
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt

DEBUG = True
"""Debug mode"""


class App(EEGStreamProcessor):
    def __init__(self, so: StreamerOptions):
        super().__init__(so)

        self.__fsamp = 256
        """Sampling rate in Hz"""
        self.__nchan = 8
        """Number of channel"""
        self.__plot_data = []
        """Plot data. (nchannel x fsamp)"""
        self.__plot_t = []
        """Time for plot"""
        self.__lines = []
        self.__st_plot = None

    def start(self):
        """Start streamer"""

        start_time = time.time()
        heart_beat = time.time()
        stream_delay = 0.8 / self.streaming_rate
        stream_count = 1

        nsamp = max(1, int(self.max_stream_duration * self.streaming_rate))
        for isamp in range(0, nsamp):

            print(f'Cycle: {isamp}/{nsamp}') if DEBUG else None
            self.read_in()
            self.update_plot()
            self.preprocess()
            self.predict()
            self.publish()

            stream_delay, stream_count, heart_beat = sleep_and_sync(
                stream_delay, stream_count, heart_beat,
                self.streaming_rate, 1/self.streaming_rate, DEBUG=DEBUG)

            # too long, shut down
            if time.time() - start_time > self.max_stream_duration:
                break

    def update_plot(self):
        sq = self._streamqueue

        timestamps, data = [], []
        for msg in [sq[i] for i in range(0, len(sq))]:
            time, values = decode(msg.key(), msg.value())
            timestamps.append(time) if time is not None else None
            data.append(values) if time is not None else None

        if len(data) < 1:
            return None
        #Update plot data and time
        self.__plot_data.extend(data)
        self.__plot_data = self.__plot_data[-int(self.__fsamp):]
        self.__plot_t.extend(timestamps)
        self.__plot_t = self.__plot_t[-int(self.__fsamp):]

        data = tuple(zip(*self.__plot_data))
        for ch in range(0, min(self.__nchan, len(data))):
            channel = data[ch]
            ydata = channel[0:min(self.__fsamp, len(channel))]

            # scale to 0 and 1
            ydata = (ydata - np.mean(ydata))/(np.max(ydata) - np.min(ydata)) + ch

            # add nan to the head
            n = self.__fsamp - len(ydata)
            ydata = np.concatenate([[np.nan]*n, ydata])

            self.__lines[ch].set_ydata(ydata)
        self.__st_plot.pyplot(plt)

    def title(self):
        st.title('SeizureCast')
        st.markdown("""
        Real-time forecasting epileptic seizure from electroencephalogram.
        """)

    def box_sampling_rate(self):
        # Sampling rate
        fsamp = st.text_input('Sampling rate (Hz)', value='256')
        try:
            fsamp = int(fsamp)
            if fsamp <= 0:
                raise Exception
        except:
            st.write('Samping rate must be positive integer')
        self.__fsamp = fsamp

    def box_file_upload(self):
        # File Upload
        uploaded_file = st.file_uploader("Choose a CSV file",
                                         type=["txt", "csv"])

        if uploaded_file is not None:
            ds = np.loadtxt(uploaded_file, delimiter=',')

            X = bin_power_avg([ds], fsamp=self.__fsamp)
            with open('model.pkl', 'rb') as fp:
                clf = pickle.load(fp)
            res = clf.predict(X)[0]
            st.write(res)

    def init_plot(self):
        """Initialize plot"""
        fig, ax = plt.subplots()
        ax.set_ylim(0, self.__nchan*1.2)
        ax.set_xlim(0, self.__fsamp)

        x = np.arange(0, self.__fsamp)
        for ch in range(0, self.__nchan):
            y = [np.nan] * len(x)
            line, = ax.plot(x, y)
            self.__lines.append(line)
        self.__st_plot = st.pyplot(plt)


    # def demo_plot(self):
    #     """Demo of real time updating plot"""
    #     fig, ax = plt.subplots()
    #
    #     x = self.__plot_t
    #     Y = self.__plot_data
    #     yscale = 0.1
    #
    #     # max_x = 5
    #     # max_rand = 10
    #
    #     # x = np.arange(0, max_x)
    #     ax.set_ylim(0, self.__nchan*1.2)
    #
    #     for ch in range(0, self.__nchan):
    #         y = Y[ch] * yscale
    #         line, = ax.plot(x, y)
    #
    #     the_plot = st.pyplot(plt)
    #
    #     def init():  # give a clean slate to start
    #         line.set_ydata([np.nan] * len(x))
    #
    #     def animate(i):  # update the y values (every 1000ms)
    #         line.set_ydata(np.random.randint(0, max_rand, max_x))
    #         the_plot.pyplot(plt)
    #
    #     init()
    #     for i in range(100):
    #         animate(i)
    #         time.sleep(0.1)


#
# def plot_eeg(data, fsamp=1):
#     """Make a plot of eeg data
#
#     Args:
#         data: nchannel x nsample
#         fsamp: samping rate. must be integer
#
#     Returns:
#         plot
#
#     """

app = App(StreamerOptions())
app.title()
app.box_sampling_rate()
app.init_plot()
app.start()
app.stop()








