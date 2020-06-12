"""
make sure you have started zookeeper server and kafka broker server
Step 1 and Step 2 in https://kafka.apache.org/quickstart
0. download kafka from apache.org, unzip using ```tar -xvf XXX```
1. run ```bin/zookeeper-server-start.sh config/zookeeper.properties```
2. run ```bin/kafka-server-start.sh config/server.properties```
"""
import ast
from confluent_kafka import Producer, Consumer, KafkaError
from time import time, sleep
from datetime import datetime
import numpy as np

KALFK_BROKER_ADDRESS = '3.130.148.125:9092'  # TODO: Implement this
STREAMER_TOPIC = 'eegstream'
CONSUMER_TOPIC = 'alert'
DEBUG = 1
"""Debug mode"""

# # asynchronous writes
# def acked(err, msg):
#     if err is not None:
#         print(f"Failed to deliver message: {str(msg)}: {str(err)}")
#     else:
#         print(f"Message produced: {str(msg)}")


def decode(key, value):
    """decode a message key and value and return list"""
    # TODO: if key is invalid then return None
    mydata = ast.literal_eval(value.decode("UTF-8"))
    return mydata['t'], mydata['v']


def sleep_and_sync(sampling_delay, sampling_count, wall_clock,
                   samping_rate, delay_refresh_intv, DEBUG=True):
    """Sleep and adjust the sampling delay

    Returns:
        list(float, int, float): sampling_delay, sampling_count,
            wall_clock
    """
    # Adhere to sampling frequency
    sleep(sampling_delay)
    sampling_count += 1

    # Adjust the sleeping interval every refresh_delay_interval seconds
    if sampling_count == (delay_refresh_intv * samping_rate):

        new_heartbeat = time()
        duration = new_heartbeat - wall_clock
        deviation = (delay_refresh_intv - duration) * 1000

        try:
            sampling_delay = sampling_delay + deviation / (
                    delay_refresh_intv * 1000) / samping_rate * 0.5
            # 0.5 = dampening factor
            if sampling_delay < 0:
                raise ValueError
        except ValueError:
            sampling_delay = 0
            print("WARNING: NEW DELAY TIME INTERVAL WAS A NEGATIVE NUMBER. Setting to 0..")
        print(f"Deviation: {deviation:.2f} ms, new delay:"
              f" {sampling_delay * 1000:.2f} ms.") if DEBUG else None
        sampling_count = 0
        wall_clock = new_heartbeat

    return sampling_delay, sampling_count, wall_clock


# class StreamerOptions:
#     def __init__(self):
#         self.max_stream_duration = 1000
#         """maximum duration in seconds"""
#         self.streaming_rate = 1
#         """streaming rate in Hz"""
#         self.delay_refresh_intv = 1.0
#         """refresh interval in seconds"""
#         self.data_shape = [22, 4]  # nchannel x nsamples
#         """shape of data"""
#         self.sampling_rate = 40
#         """data sampling rate in Hz"""


class EEGStreamer:
    """"""
    def __init__(self):
        # Initialize a Kafka Producer and a consumer
        self.producer = Producer({'bootstrap.servers': KALFK_BROKER_ADDRESS})
        """producer that produce stream of EEG signal"""

        self.producer_topic = STREAMER_TOPIC
        """producer_topic of streaming"""

        self.consumer = Consumer({
                'bootstrap.servers': KALFK_BROKER_ADDRESS,
                'auto.offset.reset': 'earliest',
                'group.id': 'group',
                'client.id': 'client',
                'enable.auto.commit': True,
                'session.timeout.ms': 6000
        })
        """consumer that subscribe the predicted results. Detailed configs 
        can be found https://towardsdatascience.com/kafka-python-explained-in-10-lines-of-code-800e3e07dad1
        """

        self.consumer_topic = CONSUMER_TOPIC
        """topic for consumer"""

        # Streaming related configs
        self.max_stream_duration = 1000
        """streaming duration in second"""
        self.streaming_rate = 4
        """Streaming rate in Hz"""
        self.delay_refresh_intv = 1.0  # time in second
        """refresh interval in seconds"""
        self.flush_interval = 0.1
        """Flush interval for Producer. In second"""

        # Listening related configs
        self.listen_interval = 1
        """Interval in second of listening"""

        # Data related configs
        self.sampling_rate = 400
        """samping rate in Hz"""
        self.nchannel = 22
        """number of channels"""
        self.montage = '1020'
        """standard 10-20 montage"""

    def start(self):
        """Start streaming.
        """
        self.consumer.subscribe([self.consumer_topic])

        # read in txt files, fake a stream data
        txt_file = './data/svdemo-pre-1.txt'
        ds = np.tile(np.loadtxt(txt_file, delimiter=','), [1, 10])
        montage = self.montage

        start_time = time()
        heart_beat = time()
        stream_delay = 0.8 / self.streaming_rate
        stream_count = 1

        for istream in range(0, ds.shape[1]):

            # Produce signal as stream
            joint_str = ','.join(['%.6f'%float(ds[ich,istream]) for ich in
                                  range(0, np.shape(ds)[0])])
            timestamp = time()
            value = "{'t':%.6f,'v':["%float(timestamp)+joint_str+"]}"
            self.producer.produce(self.producer_topic, key=montage, value=value)

            # Flush
            intv = int(self.flush_interval * self.streaming_rate)
            if stream_count % max(intv, 1) == 0:
                self.producer.flush(1)

            # Listen
            intv = int(self.listen_interval * self.streaming_rate)
            if (stream_count) % max(intv, 1) == 0:
                self.listen()

            stream_delay, stream_count, heart_beat = sleep_and_sync(
                stream_delay, stream_count, heart_beat,
                self.streaming_rate, self.delay_refresh_intv, DEBUG=True)

            # too long, shut down
            if time() - start_time > self.max_stream_duration:
                break

    def listen(self):
        """Listen to remote prediction result

        Returns:
            int: 1=alert. 0=background. 2=not sync.
        """
        # if listen == 1:
        #     print(timestamp, "!!!!!!SEIZURE IS COMING!!!!!!")
        # elif listen == 2:
        #     print(timestamp, "not sync'ed. check network connection.")
        # else:
        #     print(timestamp, "all good")
        msg = self.consumer.poll(0.1)
        if msg is None:
            pass
        elif not msg.error():
            print('Received message: {0}'.format(msg.value())) if DEBUG else None
            # TODO: if msg.key() == XXX
            t, v = decode(msg.key(), msg.value())
            t = datetime.fromtimestamp(int(t))
            # TODO: Check time stamp
            if v[0] == 'pres':
                print(t, "!!!!!!SEIZURE IS COMING!!!!!!")
            elif v[0] == 'bckg':
                print(t, "all good")
            else:
                print(t, f'UNKNOWN: {v[0]} not recognized')
        elif msg.error().code() == KafkaError._PARTITION_EOF:
            print('End of partition reached {0}/{1}'\
                  .format(msg.topic(),msg.partition())) if DEBUG else None
        else:
            print('Error occured: {0}'.format(msg.error().str())) if DEBUG else None

    def stop(self):
        # consume the remaining msg and stop the consumer
        self.consumer.consume(num_messages=10)
        self.consumer.close()
        pass


if __name__ == '__main__':
    eegstreamer = EEGStreamer()
    eegstreamer.start()
    eegstreamer.stop()
