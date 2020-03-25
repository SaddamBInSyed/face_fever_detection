
## Install and Run

1. clone repo:
```
git clone https://github.com/moshes7/face_fever_detection.git
cd face_fever_detection
```
2. create virtual env:
```
virtualenv venv_27 --python=python2
source venve_27/bin/activate

```
3. Install MXNet (.whl) with GPU support:
- Download form [drive](https://drive.google.com/open?id=1i-wgDa8rVv-9l-iR8iEhWNSLt7A9bRwZ)) the wheel file to ``mxnet_whl_path``
- install mxnet:
```
pip install mxnet_whl_path
```
4. Download Pretrained Model: RetinaFace-R50 from [drive](https://drive.google.com/open?id=1p75bDFzOa0LFTdgBJcq3BZDKOMpZ0qp4)) in place in
```
examples/RetinaFace/models
```
5. Run from ```examples/python```
```
sudo python  capture_opencv.py /ipc0
```

