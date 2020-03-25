import signal
import sys

import pyipccap

import argparse

from time import sleep


PY_MAJOR_VERSION = sys.version_info[0]

if PY_MAJOR_VERSION > 2:
    NULL_CHAR = 0
else:
    NULL_CHAR = '\0'


def main():

    g_dumpFiles = False
    g_log2Length = 2
    

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('name', type=str,
                        help='part of shared memory name')

    parser.add_argument('-d', action='store_true',
                        help='Specify to dump the incoming frames to files')

    # Optional argument
    parser.add_argument('-l', type=int,
                        help='The log 2 of number of ipc buffers to allocate.\nValid values are in the range of 0 for one buffer and 4 for 16 buffers.\nThe default is 2.')

    args = parser.parse_args()

    name = args.name
    g_dumpFiles = args.d
    g_log2Length = args.l

    if g_log2Length < 0:
        print("Log 2 Length of %d is less than 0. 0 will be used\n", g_log2Length)
        g_log2Length = 2

    if g_log2Length > 4:
        print("Log 2 Length of %d is greater than 4. 4 will be used\n", g_log2Length)
        g_log2Length = 4;

    print("g_log2Length %d\n", g_log2Length)

    nOutputImageQ = 1 << g_log2Length

    print "name ", name

    print "nOutputImageQ ", nOutputImageQ
    
    thermapp = pyipccap.thermapp(name,nOutputImageQ)
    
    print "Got frame " , g_dumpFiles

    thermapp.open_shared_memory()
    
    while True:

        data = thermapp.get_data()
        if data is None :
            continue
		
		#print "saved image \n"
		'''
		frame_name = "/home/debian/opgal/dump_python/frame_{}.bin".format(thermapp.imageId)
		f = open(frame_name, "w+")
		f.write(data)
		f.close()
		'''




if __name__ == '__main__':
    main()
