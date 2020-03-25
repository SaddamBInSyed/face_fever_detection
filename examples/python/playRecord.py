print("Loading...")
import sys, os
import cv2
import numpy as np
import os
import glob
import argparse



def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', type=str,
                        help='input path')
    args = parser.parse_args()
    recordPath = args.i
    
    if recordPath is None:
        hardcoded_default_location = "/media/nano/Elements/"
        folders_list=glob.glob(os.path.join(hardcoded_default_location,"*"))
        if len(folders_list) == 0:
            print("ERROR: there are 0 folders in default location {}".format(hardcoded_default_location))
            exit()
        recordPath = sorted(folders_list)[-1] # Take the last existing folder

    

    file_list = glob.glob(os.path.join(recordPath, "*.bin"))
    file_list.sort()
    
    if len(file_list) is 0:
        print("Folder {} is empty. Nothing to play :(".format(recordPath))
        exit()
    print("found {} files, start playing...".format(len(file_list)))
    counter = 0
    try:
        for file in file_list:
            print("{}: {}".format(counter, os.path.basename(file)))
        
            off=64//2
            #im=np.frombuffer(open(sys.argv[1],'rb').read(),'uint8')[off:640*480+off].reshape((480,640))
            im=np.frombuffer(open(file,'rb').read(),'int16')[off:384*288+off].reshape((288,384))
            im=im-im.min()
            im=im.astype('float')
            im=(im/(im.max())*255).astype('uint8')
            #print(im.max(),im.min())
            cv2.imshow("show",im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter += 1

    except KeyboardInterrupt:
        print("Caught ya!")

    print("That's all Folks!")



if __name__ == '__main__':
    main()
