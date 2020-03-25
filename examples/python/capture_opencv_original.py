import signal
import sys
import time
import pyipccap
import os
import datetime
import argparse

import numpy as np

import cv2

from time import sleep


PY_MAJOR_VERSION = sys.version_info[0]

if PY_MAJOR_VERSION > 2:
    NULL_CHAR = 0
else:
    NULL_CHAR = '\0'


def main():
   

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('name', type=str,
                        help='part of shared memory name')

    parser.add_argument('-noshow', action='store_true', default = False,
                        help='Specify -nowshow flag if you don\'t want to show the images on the screen')
    
    parser.add_argument('-d', action='store_true', default = False,
                        help='Specify to dump the incoming frames to files')

    parser.add_argument('-o', type=str, default="/media/nano/Elements/",
                        help='if -d is specified => will save images to the specified directory')

    # Optional argument
    parser.add_argument('-l', type=int,
                        help='The log 2 of number of ipc buffers to allocate.\nValid values are in the range of 0 for one buffer and 4 for 16 buffers.\nThe default is 2.')

    args = parser.parse_args()

    name = args.name
    g_dumpFiles = args.d
    g_log2Length = args.l
    g_showPic = not args.noshow
    g_dumpPath = args.o
    g_drawGraph = True # showing the pic implies not drawing the graph
    
    print("Params:")
    print("name:          {}".format(name))
    print("Dump files:    {}".format(g_dumpFiles))
    if g_dumpFiles:
        g_dumpPath = os.path.join(g_dumpPath, datetime.datetime.now().strftime('%Y_%d_%m__%H_%M_%S'))
        print("Dump path      {}".format(g_dumpPath))
        try:
            os.mkdir(g_dumpPath)
        except:
            import traceback; traceback.print_exc()
            print("ERROR: can not create folder: {}".format(g_dumpPath))
            import pdb;pdb.set_trace()
    print("Show image:    {}".format(g_showPic))

    g_log2Length = 2

    if g_log2Length < 0:
        print("Log 2 Length of %d is less than 0. 0 will be used\n", g_log2Length)
        g_log2Length = 2

    if g_log2Length > 4:
        print("Log 2 Length of %d is greater than 4. 4 will be used\n", g_log2Length)
        g_log2Length = 4;

    print("g_log2Length:  {}".format(g_log2Length))

    nOutputImageQ = 1 << g_log2Length
    print("nOutputImageQ: {} ".format(nOutputImageQ))


    thermapp = pyipccap.thermapp(None,nOutputImageQ)
    #thermapp = pyipccap.thermapp(name,nOutputImageQ)
    
    print "Open shared memory..."

    thermapp.open_shared_memory()
    
    print "Shared memory opened"
    
    rate_list = []
    frame_count = 0
    prev = 0
    last_msg = time.time()
    avg_accum = 0
    avg_count = 0
    total_misses = 0
    try:
        while True:

            data = thermapp.get_data()
            if data is None :
                continue
        
            # We have new data

            # time calc
            now = time.time()
            diff = now - last_msg
            last_msg = now
            if diff > 0:
                msg_rate = 1.0/diff
            else:
                msg_rate = 0.0
            miss = thermapp.imageId - prev - 1
            prev = thermapp.imageId
        

            avg_max = 60 # count avg every avg_max frames
            if avg_count < avg_max:
                avg_accum += msg_rate
                avg_count += 1
                print("Got {}, dim ({},{}), missed = {}, rate = {} Hz".format(thermapp.imageId, thermapp.imageWidth, thermapp.imageHeight, miss, round(msg_rate,2)))
            else:
                avg_rate = avg_accum/avg_max
                print("Got {}, dim ({},{}), missed = {}, rate = {} Hz, avg_rate = {} Hz".format(thermapp.imageId, thermapp.imageWidth, thermapp.imageHeight, miss, round(msg_rate,2), round(avg_rate,2)))
                avg_accum = 0
                avg_count = 0
            
            if (frame_count) > 10:
                rate_list.append(msg_rate)
                total_misses += miss

            if g_showPic:

                img_st = data[64:]
                rgb = np.frombuffer(img_st, dtype=np.uint16).reshape(thermapp.imageHeight, thermapp.imageWidth)
                #rgb = np.uint8(rgb)

                im = rgb - rgb.min()
                im = im.astype('float')
                im = (im/(im.max())*255).astype('uint8')
                
                cv2.imshow('image', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if g_dumpFiles:
                frame_name = os.path.join(g_dumpPath, "frame_{}.bin".format(thermapp.imageId))
                print("save image: {}".format(frame_name))
                
                f = open(frame_name, "w+")
                f.write(data)
                f.close()
                

            frame_count += 1 # captured frames counter

    except KeyboardInterrupt:
        print("Caught ya!")

    if g_drawGraph and not g_showPic:
        print("Finished. Now drawing the graph")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1)
        ax.plot(rate_list, label='Rate[Hz]')
        avg = sum(rate_list)/len(rate_list)
        ax.axhline(y = avg, color='r', linestyle='-', label='AVG='+str(round(avg,2)))
        ax.plot([], color='r', label='Misses='+str(total_misses))
        ax.legend(loc='uppder left')
        ax.set_xlabel('time')
        ax.set_ylabel('rate[Hz]')
        plt.show()



if __name__ == '__main__':
    main()
