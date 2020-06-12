# Deploy streamlit on EC2
1. Setup EC2 running ubuntu 18.04
Follow instructions here: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html

* Set up security group to allow streamlit port 8501 visted from private IP.

2. Git clone your app (github repo) to the server

* Install Streamlit *using sudo*
```sudo python3 -m pip install streamlit```
* Test the streamlit is installed properly. The following line should pass.
```streamlit -h```

3. Install tmux and run a session.
* new session: 
`tmux new -s my_session`

* run streamlit app: 
`streamlit run app.py`

* detach session by key sequence Ctrl-b + d to detach from the session

* (optional) kill all sessoins:
```tmux kill-session -a```

4. Install nginx and 

* Run
``` sudo vim /etc/nginx/sites-enabled/streamlit```

* Add the following (Change `YOUR_DOMAIN_NAME_NO_SLASH` to your domain name)
```
server {
        listen 80;
        server_name YOUR_DOMAIN_NAME_NO_SLASH;  
        location /google {
                proxy_pass http://google.com/;
        }
        location / { # most important config
                proxy_pass http://localhost:8501/;
                proxy_http_version 1.1; 
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header Host $host;
                proxy_set_header Upgrade $http_upgrade;
                proxy_set_header Connection "upgrade";
                proxy_read_timeout 86400;
        }
}
```

And save it by `:wq`

* Reload nginx configures.
```
sudo nginx -s reload
```

5. Try it out.

Test `YOUR_DOMAIN_NAME/google` should point to google.com.

Test `YOUR_DOMAIN_NAME` should point to your app

# Running Kafka on EC2

The real-time modeling will be done by EEGStreamProcessor.py.

REF: https://dzone.com/articles/installing-and-running-kafka-on-an-aws-instance

1. Setting up EC2 security group.

2. Installing Apache Kafka
3. Running Zookeeper

Running the following srvr command, should return a list of the server details.
```echo srvr | nc localhost 2181```
Note: 3.5.3-beta has all four letter words disabled except srvr, please refer to the doc here: http://zookeeper.apache.org/doc/r3.5.3-beta/zookeeperAdmin.html

4. Setting up Kafka

Add the following line to config/server.properties 
```advertised.listeners=PLAINTEXT://<your.public.IP>:9092```

Start Kafka Server. First without the ```-daemon``` and make sure the server started. Then run the following.
```
bin/kafka-server-start.sh -daemon config/server.properties
```

Test Kafka Server is up and running.
```
bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic testTopic
```
Test producer is working by run the following lines and send any messages.
```
bin/kafka-console-producer.sh --broker-list localhost:9092 --topic testTopic
```
Test consumer is working.
```
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic testTopic --from-beginning
```

5. Test on Python script
Broker address is YOUR_IP:9092

6. Kafka & EC2 references
REF1: https://stackoverflow.com/questions/43565698/connecting-kafka-running-on-ec2-machine-from-my-local-machine
REF2: https://medium.com/@maftabali2k13/setting-up-a-kafka-cluster-on-ec2-1b37144cb4e



