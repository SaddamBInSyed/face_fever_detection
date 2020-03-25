
## Install and Run

1. clone repo:
```
git clone https://github.com/moshes7/face_fever_detection.git
cd face_fever_detection
```
2. create virtual env and install pkg:
```
virtualenv venv_27 --python=python2
source venv_27/bin/activate
pip install -r requirements.txt
```
3. Install MXNet (.whl) with GPU support:
- Download from [drive](https://drive.google.com/open?id=1i-wgDa8rVv-9l-iR8iEhWNSLt7A9bRwZ) the wheel file to ``mxnet_whl_path``
- install mxnet:
```
pip install mxnet_whl_path
```
Download from [drive](https://drive.google.com/open?id=1fPVmW7MVMW1C13mv0ZEDXv1JmHRyZzbh) and replace the file ``libmxnet.so`` in ``venv_27/lib/python2.7/site-packages/mxnet``

4. Download Pretrained Model: RetinaFace-R50 from [drive](https://drive.google.com/open?id=1p75bDFzOa0LFTdgBJcq3BZDKOMpZ0qp4) and place in
```
examples/RetinaFace/models
```
5. Compile additional source:
```
cd /examples/RetinaFace
make
```
**** make sure to copy the cv2 and cv2.so from native python (``/usr/lib/python2.7/dist-packages``) to ``venv_27/lib/python2.7/site-packages/`` ****
6. open new terminal window and run:
```
sudo bin/erop-proxy-cam -nproxycam-ir --perror --logmask=130
```
7. Run from ```examples/python```:
```
 sudo ../../venv_27/bin/python  capture_opencv.py /ipc0
```

