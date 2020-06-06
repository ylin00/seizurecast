"""
This script retrieves Kafka EEG stream and produces predicted results

Author:
    Yanxian Lin, Insight Health Data Science Fellow, Boston 2020
"""

from EEGStreamer import KALFK_BROKER_ADDRESS, CONSUMER_TOPIC, STREAMER_TOPIC, sleep_and_sync
from confluent_kafka import Producer, Consumer, KafkaError
from time import time, sleep


class EEGStreamProcessor:
    def __init__(self):
        self.consumer = Consumer({
                'bootstrap.servers': KALFK_BROKER_ADDRESS,
                'auto.offset.reset': 'earliest',
                'group.id': 'group',
                'client.id': 'client',
                'enable.auto.commit': True,
                'session.timeout.ms': 6000
        })
        """consumer that reads stream of EEG signal"""
        self.producer = Producer({'bootstrap.servers': KALFK_BROKER_ADDRESS})
        """producer that produces predition results"""

        self.samping_rate = 400
        """processing rate in Hz"""

        self.delay_refresh_intv = 1.0  # time in second
        """refresh interval in seconds"""

        self.max_stream_duration = 1000
        """maximum duration in seconds"""

        self.data_shape = [22, 400]  # nchannel x nsamples
        """shape of data"""

        self.__streamqueue = []
        """queue for stream"""

        self.__dataqueue = []
        """queue for data"""

        self.__predicted_result = None
        """"""

    def start(self):
        self.consumer.subscribe([STREAMER_TOPIC])

        start_time = time()
        wall_clock = time()
        sampling_delay = 0.8 / self.samping_rate
        sampling_count = 1

        for isamp in range(0, max(1, int(self.max_stream_duration))):

            self.read_in()
            self.preprocess()
            self.predict()
            self.publish()

            sampling_delay, sampling_count, wall_clock = sleep_and_sync(
                sampling_delay, sampling_count, wall_clock, self.samping_rate,
                self.delay_refresh_intv, DEBUG=True)

            # too long, shut down
            if time() - start_time > self.max_stream_duration:
                break

    def read_in(self):
        """read stream from Kafka

        Returns:
            list of list: dataset (nchannel x nsample) or None
        """
        msgs = self.consumer.consume(num_messages=self.samping_rate)
        if msgs is not None and len(msgs) > 0:
            print('I got some msgs!')
            # TODO:
        raise NotImplementedError

    def preprocess(self):
        """preprocess data"""
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def publish(self):
        """publish predicted result"""
        raise NotImplementedError

    def stop(self):
        self.consumer.close()
        pass


if __name__ == '__main__':
    esp = EEGStreamProcessor()
    esp.start()
    esp.stop()
