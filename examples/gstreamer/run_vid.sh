#!/bin/bash
gst-launch-1.0 -v udpsrc port=5555 ! application/x-rtp,clock-rate=90000,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink
