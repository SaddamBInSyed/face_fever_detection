#!/usr/bin/env python
# to open the mkv 
#mkdir data
#gst-launch-1.0 -m filesrc location=video.mkv ! decodebin ! videoconvert ! jpegenc ! multifilesink location=data/frame%08d.jpg post-messages=true > timings.txt


import sys
import time
import numpy as np
import traceback
from ipdb import set_trace as st
import gi
#gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
nvidia=True
use_yuy2=True

def get_dims(pipeline,name='dec'):
    pad = pipeline.get_by_name(name).pads[0]
    caps=pad.get_current_caps()
    struct=caps.get_structure(0)
    h=struct.get_int('height')[1]
    w=struct.get_int('width')[1]
    if h==0:
        print('Error: dim h is 0')
        return None
    return (h,w,3)

def get_image_from_sink(sink,dims):
    sample=sink.emit('pull-sample') 
    buf=sample.get_buffer()
    mem=buf.get_all_memory()
    ret,mi=mem.map(Gst.MapFlags.READ)
    #t = (self.pipeline.clock.get_time()-self.pipeline.base_time)/1e9
    t = (sink.clock.get_time()-sink.base_time)/1e9
    if use_yuy2:
        image=(t,np.frombuffer(mi.data,'uint8'))
    else:
        image=(t,np.frombuffer(mi.data,'uint8').reshape(dims))
    mi.memory.unmap(mi)
    return image
     


class Reader:   
    def on_new_buffer(self,sink):
        if self.dims is None:
           self.dims=get_dims(self.pipeline)
        self.image=get_image_from_sink(sink,self.dims)
        return Gst.FlowReturn.OK

    def __init__(self,fname):
        Gst.init(None)
        #pipe = 'filesrc location=out.mkv ! matroskademux ! avdec_h265 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink'
        pipe = 'filesrc location={} ! matroskademux ! decodebin name=dec ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink'.format(fname)
        pipeline = Gst.parse_launch(pipe)

        sink=pipeline.get_by_name('sink')
        sink.set_property("max-buffers",2)
        sink.set_property("emit-signals", True)
        sink.connect("new-sample", self.on_new_buffer) 
        pipeline.set_state(Gst.State.READY)
        self.pipeline=pipeline
        self.image=None
        self.pipeline.set_state(Gst.State.PLAYING)
        self.dims=None

    def get_next(self):
        bus = self.pipeline.get_bus()
        while self.image is None:
            message = bus.timed_pop_filtered(3*Gst.MSECOND,Gst.MessageType.ANY)
            if message:
                t = message.type
                if t == Gst.MessageType.EOS:
                    sys.stdout.write("End-of-stream\n")
                    break
                elif t == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    sys.stderr.write("Error: %s: %s\n" % (err, debug))
                    break
        # self.pipeline.set_state(Gst.State.PAUSED)
        ret=self.image
        self.image=None
        return ret
    def pause(self):
        self.pipeline.set_state(Gst.State.PAUSED) 
    def play(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def __del__(self):    # cleanup
        self.pipeline.set_state(Gst.State.NULL)


class Writer:
    def __init__(self,fname,ip='127.0.0.1',port=6666):
        Gst.init(None)
        # pipe = 'videotestsrc is-live=true horizontal-speed=5 name=src ! x265enc bitrate=500 ! h265parse ! matroskamux ! filesink location=out.mkv'
        # pipe = 'videotestsrc is-live=true horizontal-speed=5 name=src '
        #pipe = 'v4l2src device=/dev/video2 name=src ! video/x-raw,format=YUY2,width=640,height=480 ' 

        #based on:
        #send gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM), width=4056,height=3040, framerate=30/1' ! nvvidconv flip-method=0 ! 'video/x-raw, format=(string)I420, width=(int)640, height=(int)480' ! omxh264enc control-rate=2 bitrate=1500000 ! video/x-h264, stream-format=byte-stream !  rtph264pay ! udpsink port=5000 host=10.42.134.19 sync=false

        #recive gst-launch-1.0 -v udpsrc port=5000 ! application/x-rtp,clock-rate=90000,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink
		
        self.dims = (4056, 3040)

        enc='264' 
        bitrate=1500
        if nvidia:
            tsw,tsh=self.dims[0]//4, self.dims[1]//4 # 4056,3040 //4  IMX477 sensor
            encline1='! omxh{0}enc control-rate=2 bitrate={1} ! video/x-h{0}, stream-format=avc '.format(enc,bitrate*1000*10)
            encline2='! omxh{0}enc control-rate=2 bitrate={1} ! video/x-h{0}, stream-format=byte-stream '.format(enc,bitrate*1000)
            #pipe = 'v4l2src device=/dev/video2 name=src ! video/x-raw,format=YUY2,width=640,height=480 '
            pipe = 'nvarguscamerasrc sensor-id=0 name=src ! video/x-raw(memory:NVMM),width=4056,height=3040,framerate=30/1 '
            pipe += '! tee name=t ! queue '
            pipe += '! nvvidconv ! video/x-raw,format=YUY2 ! appsink name=sink '
            pipe += 't. ! queue ! '

            ##pipe += 'videoconvert ! video/x-raw,format=I420 '
            pipe += "nvvidconv ! video/x-raw(memory:NVMM),width=(int){},height=(int){},format=(string)I420 ".format(tsw*2,tsh*2)
            pipe += encline1
            pipe += '! h{0}parse ! matroskamux name=dec ! filesink location={1} sync=false '.format(enc,fname)
            ##pipe += '! fakesink sync=false '.format(enc)
            ##pipe += '! filesink location=out.h{} '.format(enc)
            ##pipe += '! filesink location=out.h{} '.format(enc)
            pipe += 't. ! queue ! '
            ##pipe += 'videoconvert ! video/x-raw,format=I420 '
            pipe += "nvvidconv ! video/x-raw(memory:NVMM),width=(int){},height=(int){},format=(string)I420 ".format(tsw,tsh)
            pipe += encline2
            pipe += '! rtph{}pay ! udpsink port={} host={} sync=false'.format(enc,port,ip)
        
        else:
            encline='! x{}enc tune=zerolatency  bitrate={} '.format(enc,bitrate)
            pipe = 'v4l2src device=/dev/video2 name=src ! video/x-raw,format=YUY2,width=384,height=288 '
            pipe += '! tee name=tt ! queue ! '
            pipe += 'videoconvert ! video/x-raw,format=RGB ! appsink name=sink '
            pipe += 'tt. ! queue ! '

            pipe += 'videoconvert ! video/x-raw,format=I420 '
            pipe += encline
            pipe += '! tee name=t ! queue ! '
            pipe += 'h{}parse ! matroskamux name=dec ! filesink location={} sync=false '.format(enc,fname)
            pipe += 't. ! queue ! '
            pipe += 'rtph{}pay ! udpsink port={} host={} sync=false'.format(enc,port,ip)
            


        print('-'*100)
        print(pipe)
        print('-'*100)

        self.pipeline = Gst.parse_launch(pipe)
        print('*'*100)
       

        sink=self.pipeline.get_by_name('sink')
        sink.set_property("max-buffers",2)
        sink.set_property("emit-signals", True)
        sink.connect("new-sample", self.on_new_buffer) 
        self.source = self.pipeline.get_by_name('src')
        
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PLAYING)
        
        self.dims=None
        self.image=None
    
    def on_new_buffer(self,sink):
        try:
            if self.dims is None:
                self.dims=get_dims(self.pipeline, 'src')
            self.image=get_image_from_sink(sink,self.dims)
        except Exception as e:
            print('ERROR',traceback.format_exc())
        return Gst.FlowReturn.OK

     
    def get_next(self):
        pipeline=self.pipeline
        bus = pipeline.get_bus()
        message = bus.timed_pop_filtered(5*Gst.MSECOND,Gst.MessageType.ANY)
         
        stream_time=(self.source.clock.get_internal_time()-self.source.base_time)/1e9
        if message:
            t = message.type
            if t == Gst.MessageType.EOS:
                sys.stdout.write("End-of-stream\n")
            elif t == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                sys.stderr.write("Error: %s: %s\n" % (err, debug))
        ret={'ts':stream_time}
        if self.image is not None:
            ret['image']=self.image
            self.image=None
        return ret 
    
    def __del__(self):    # cleanup
        self.pipeline.set_state(Gst.State.NULL)
 

if __name__ == '__main__':
    import argparse
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", help="is reader",default=False,action='store_true')
    parser.add_argument("--ip", help="destenation ip",default='127.0.0.1')
    parser.add_argument("--port", help="destenation ip",default=6666,type=int)
    args = parser.parse_args()
    if args.reader:
        rd = Reader('out.mkv')
        cnt=0
        while 1:
            t,img = rd.get_next()
            print('t=',t)
            if img is None:
                break
            cv2.imshow('img',img)
            if cnt==10:
                rd.pause()
                time.sleep(5)
                rd.play()
            k=cv2.waitKey(1)
            if k%256==ord('q'):
                break
            cnt+=1
    else:
        print('start writer...')
        wt = Writer('out.mkv', ip=args.ip, port=args.port)
        tic=time.time()
        dflag=True
        while time.time()-tic<135:
            #time.sleep(0.5)
            if dflag and time.time()-tic>15:
                print('write dot file if GST_DEBUG_DUMP_DOT_DIR env exist')
                #run with GST_DEBUG_DUMP_DOT_DIR=dumppath
                Gst.debug_bin_to_dot_file(wt.pipeline,Gst.DebugGraphDetails.ALL,'out')
                dflag=False
            ret = wt.get_next()
            if 'image' in ret:
                print('got image',ret['image'][0],ret['image'][1].shape,time.time()-tic)
