
## Install and Run

1. install system-level prerequisites:
```
sudo apt update
sudo apt install -y python-virtualenv
sudo apt install -y curl wget
sudo apt install -y libhdf5-dev libhdf5-serial-dev
```
2. clone repo:
```
git clone https://github.com/moshes7/face_fever_detection.git
cd face_fever_detection
```
3. create virtual env and install pkg:
```
virtualenv venv_27 --python=python2
source venv_27/bin/activate
pip install -r requirements.txt
```
4. Download (from [drive](https://drive.google.com/open?id=1i-wgDa8rVv-9l-iR8iEhWNSLt7A9bRwZ)) and install MXNet (.whl) with GPU support:
```
bash bin/get_gdrive_file.sh 1i-wgDa8rVv-9l-iR8iEhWNSLt7A9bRwZ /tmp/mxnet-1.6.0-py2-none-any.whl
pip install /tmp/mxnet-1.6.0-py2-none-any.whl
```
5. Download from [drive](https://drive.google.com/open?id=1fPVmW7MVMW1C13mv0ZEDXv1JmHRyZzbh) and replace the file ``libmxnet.so`` in ``venv_27/lib/python2.7/site-packages/mxnet``
```bash
bash bin/get_gdrive_file.sh 1fPVmW7MVMW1C13mv0ZEDXv1JmHRyZzbh venv_27/lib/python2.7/site-packages/mxnet/libmxnet.so
```
6. Download Pretrained Model: RetinaFace-R50 from [drive](https://drive.google.com/open?id=1p75bDFzOa0LFTdgBJcq3BZDKOMpZ0qp4) and place in examples/RetinaFace/models
```
mkdir -p examples/RetinaFace/models
bash bin/get_gdrive_file.sh 1p75bDFzOa0LFTdgBJcq3BZDKOMpZ0qp4 examples/RetinaFace/models/retinaface-R50.zip
unzip examples/RetinaFace/models/retinaface-R50.zip -d examples/RetinaFace/models
```
7. Compile additional source:
```
cd examples/RetinaFace
make -j
cd -
```

8. Download (from [drive](https://drive.google.com/open?id=1uCF6s-wjluZLIwb9150KMlfZ0G25UQiA)) and install opencv (including contrib and aruco)
```
bash bin/get_gdrive_file.sh 1uCF6s-wjluZLIwb9150KMlfZ0G25UQiA opencv.tar.gz
tar -xzf opencv.tar.gz
echo export LD_LIBRARY_PATH=$(readlink -f opencv/lib):$LD_LIBRARY_PATH >> ~/.bashrc
deactivate
source ~/.bashrc
source venv_27/bin/activate
```
9. Test opencv installation (the following command should return with no errors)
```
python -c "import cv2.aruco"
```
10. Copy camera configuration files:
```
sudo mkdir -p /etc/eyerop
sudo cp -rf etc/* /etc/eyerop/
```
11. open new terminal window and run:
```
sudo bin/erop-proxy-cam -nproxycam-ir --perror --logmask=130
```
12. Run from ```examples/python```:
```
 sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH ../../venv_27/bin/python  capture_opencv.py /ipc0
```

