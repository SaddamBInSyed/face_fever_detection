#!/bin/bash

if [ "$1" = "kill" ]; then
    tmux kill-server
    kill $(ps aux | grep "sudo python capture_opencv.py" | awk '{print $2}')
    echo "killed"
    exit
fi

# first kill the old server
sudo pkill erop-proxy-cam
tmux kill-server
sleep 1


tmux new-session -d
#######
#.....#
#.....#
#.....#
#######
tmux send-keys 'cd /opt/eyerop/examples/ipccap/; ./run_erop.sh' KPEnter

tmux split-window -h 
#######
#  #..#
#  #..#
#  #..#
#######

tmux split-window -v -t 0
#######
#  #  #
####  #
#..#  #
#######

# original
# tmux send-keys 'cd /opt/eyerop/examples/python/; ./run_python_cap.sh' " $*" KPEnter
tmux send-keys 'cd /home/nvidia/eyerop/eyerop-master/examples/python/; ./run_python_cap.sh' " $*" KPEnter


tmux split-window -v -t 2
#######
#  #  #
#######
#  #..#
#######
tmux send-keys 'watch -n1 --no-title df -BM /dev/shm' KPEnter

tmux select-pane -t 2
#######
#  #..#
#######
#  #  #
#######

tmux att
