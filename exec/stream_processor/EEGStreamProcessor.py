"""
EEG Stream Processor

Consuming EEG stream from Kafka and producing predicted results from given model.

Author:
    Yanxian Lin, Insight Health Data Science Fellow, Boston 2020
"""
# TODO: README
import pickle
from confluent_kafka import Producer, Consumer
from collections import deque
from time import time

# TODO: parse configs from ini files following https://docs.python.org/3/library/configparser.html
# TODO: functions as utils
from exec.streamer.EEGStreamer import KALFK_BROKER_ADDRESS, CONSUMER_TOPIC, STREAMER_TOPIC, \
    sleep_and_sync, decode
from src.features.dataset_funcs import bin_power_avg

DEBUG = True


class StreamerOptions:
    def __init__(self):

        # Streaming related configs
        self.max_stream_duration = 10000000000
        """maximum duration in seconds. How long the Streamer should run?"""
        self.streaming_rate = 10
        """streaming rate in Hz. Number of data to process per second. """
        self.delay_refresh_intv = 1/self.streaming_rate
        """refresh interval in seconds."""

        # Data related configs
        self.sampling_rate = 256
        """data sampling rate in Hz. Properties of the data. """
        self.data_shape = [8, self.sampling_rate]  # nchannel x nsamples
        """shape of data. Number of channels x sampling rate."""


class EEGStreamProcessor:

    def __init__(self, so: StreamerOptions):
        self.consumer = Consumer({
                'bootstrap.servers': KALFK_BROKER_ADDRESS,
                'auto.offset.reset': 'earliest',
                'group.id': 'group-processor',
                'client.id': 'client-local',
                'enable.auto.commit': True,
                'session.timeout.ms': 6000
        })
        """consumer that reads stream of EEG signal"""
        self.producer = Producer({'bootstrap.servers': KALFK_BROKER_ADDRESS})
        """producer that produces predition results"""

        self.streaming_rate = so.streaming_rate
        self.max_stream_duration = so.max_stream_duration
        self.data_shape = so.data_shape
        self.sampling_rate = so.sampling_rate

        self.__streamqueue = deque()
        # queue for raw data
        self.__data = deque()
        self.__data_t = deque()
        # queue of processed_data
        self.__pdata = deque()
        self.__pdata_t = deque()
        # queue of results
        self.__res = deque()
        self.__res_t = deque()
        self.__model = None

        self.setup()

    def setup(self):
        self.consumer.subscribe([STREAMER_TOPIC])
        with open('./models/model.pkl', 'rb') as fp:
            self.model = pickle.load(fp)

    def start(self):
        """Start streamer"""

        start_time = time()
        heart_beat = time()
        stream_delay = 0.8 / self.streaming_rate
        stream_count = 1

        nsamp = max(1, int(self.max_stream_duration * self.streaming_rate))
        for isamp in range(0, nsamp):

            print(f'Cycle: {isamp}/{nsamp}') if DEBUG else None
            self.read_in()
            self.preprocess()
            self.predict()
            self.publish()

            stream_delay, stream_count, heart_beat = sleep_and_sync(
                stream_delay, stream_count, heart_beat,
                self.streaming_rate, 1/self.streaming_rate, DEBUG=DEBUG)

            # too long, shut down
            if time() - start_time > self.max_stream_duration:
                break

    def read_in(self):
        """read stream from Kafka and append to streamqueue

        Returns:
            list of list: dataset (nchannel x nsample) or None
        """
        chunk_size = self.data_shape[1]
        msgs = []
        while chunk_size > 100:
            msgs.extend(self.consumer.consume(num_messages=100, timeout=1))
            chunk_size -= 100
        msgs.extend(self.consumer.consume(num_messages=chunk_size, timeout=1))

        print(f"Received {str(len(msgs))} messages") if DEBUG else None

        if msgs is None or len(msgs) <= 0:
            return None

        self.__streamqueue.extendleft(msgs)  # Enqueue

        if len(self.__streamqueue) < self.data_shape[1]:
            return None

        # Dequeue
        msgs__ = [self.__streamqueue.pop() for i in range(0, self.data_shape[1])]

        timestamps, data = [], []
        for msg in msgs__:
            time, values = decode(msg.key(), msg.value())
            timestamps.append(time) if time is not None else None
            data.append(values) if time is not None else None
        #TODO:// assert there is not big time gap in the data

        if len(data) < self.data_shape[1]:
            return None

        print(timestamps[0], data[0]) if DEBUG else None

        data = tuple(zip(*data))
        self.__data.append(data)
        self.__data_t.append(timestamps[0])

        print(f"INFO: Sucessfully Read a chunk") if DEBUG else None

    def preprocess(self):
        """preprocess data"""
        if len(self.__data) <= 0:
            self.__data.clear(), self.__data_t.clear()
            return None
        data = [self.__data.pop() for i in range(0, len(self.__data))]
        time = [self.__data_t.pop() for i in range(0, len(self.__data_t))]

        X = bin_power_avg(data, fsamp=self.sampling_rate)
        self.__pdata.extendleft(X)
        self.__pdata_t.extendleft(time)

    def predict(self):
        if len(self.__pdata) <= 0:
            return None
        for i in range(0, len(self.__pdata)):
            processed_data = self.__pdata.pop()
            processed_t = self.__pdata_t.pop()
            try:
                predicted_rels = self.model.predict([processed_data])
            except ValueError:
                return None
            self.__res.appendleft(predicted_rels[0])
            self.__res_t.appendleft(processed_t)

    def publish(self):
        """publish predicted result"""
        for i in range(0, len(self.__res)):
            res = self.__res.pop()
            tim = self.__res_t.pop()
            joint_str = res
            #Fixme: ductape the model prediction should be restricted
            if res == 0:
                joint_str = 'bckg'
            elif res == 1:
                joint_str = 'pres'
            key = 'key'
            value = "{'t':%.6f,'v':["%float(tim)+"'"+joint_str+"'"+"]}"
            self.producer.produce(CONSUMER_TOPIC, key=key, value=value)
            print(f'Published: {tim}, {res}') if DEBUG else None

    def stop(self):
        self.consumer.close()
        pass

    @property
    def _streamqueue(self):
        """Access to streamqueue"""
        return self.__streamqueue

    @property
    def _result(self):
        """Access to predicted results"""
        return self.__res


if __name__ == '__main__':
    esp = EEGStreamProcessor(StreamerOptions())
    esp.start()
    esp.stop()
