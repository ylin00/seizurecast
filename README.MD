# Deploy streamlit on EC2
1. Setup EC2 running ubuntu 18.04
Follow instructions here: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html

2. Git clone your app (github repo) to the server

2.5. Install Streamlit *using sudo*
```sudo python3 -m pip install streamlit```
Check the following: 
```streamlit -h```

3. Install tmux and run a session.\\
3.1. new session: 
`tmux new -s my_session`

3.2. run streamlit app: 
`streamlit run app.py`

3.3. detach session by key sequence Ctrl-b + d to detach from the session

3.4. (optional) kill all sessoins:
```tmux kill-session -a```

4. Install nginx and 

4.1.
``` sudo vim /etc/nginx/sites-enabled/streamlit```

Add the following
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

4.2. Reload nginx configures.
```
sudo nginx -s reload
```

5. Try it out.

Test `YOUR_DOMAIN_NAME/google` should point to google.com.

Test `YOUR_DOMAIN_NAME` should point to your app

