# run tmux (grabber+capturer)
---->
sudo /opt/eyerop/examples/run_thermapp.sh (-noshow) (-d) (-i /media/nano/Elements/)

# run player (w/o args plays last folder in /media/nano/Elements/)
---->
/opt/eyerop/examples/run_player.sh (-i /media/nano/Elements/2019_26_12__21_07_35)



git:
https://github.com/serbuh/eyerop.git

# run frame grabber (first)
/opt/eyerop/examples/ipccap/run_erop.sh

# run frame capture - python version
/opt/eyerop/examples/python/run_python_cap.sh

sudo python /opt/eyerop/examples/python/capture_opencv.py /ipc0
sudo python /opt/eyerop/examples/python/capture_opencv.py /ipc0 -d

# kill python frame capturer
sudo pkill python

# compile python code
/opt/eyerop/examples/ipccap/compile_python.sh

# run frame capture (c)
nano@jetson-nano:/opt/eyerop/examples/ipccap$ sudo ./ipccap /ipc0 -d

# show one frame with opencv
python /opt/eyerop/examples/showcv.py '/media/nano/Elements/frame_1211.bin'  

