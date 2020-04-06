# python client server rest API example

## installation
1. install python3-virtualenv
```
sudo apt install python3-virtualenv
```
2. create python3 virtual environment, activate it and install packages:
```bash
virtualenv -p python3 venv
. venv/bin/activate
pip install -r requirements.txt
```
## run the server
1. activate the virtual env (if not active) and launch the server app
```
. venv/bin/activate
python rest_api_server.py
```
## run the client
1. on the same host open a new terminal window and activate the virtual env
```
. venv/bin/activate
```
2. you may repeatedly run the client program to send the data to the client and parse the response: 
```
python post_image_example.py
```