# This example demonstrate how to use erop-tunnel to get the properties
# of the running pipeline in erop-proxy-cam

# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import json
import getopt
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

def perror(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def send_erop_command(sysname, target, commandName, commandData):
    import tempfile
    retVal = ""
    commandString = json.dumps({
                                u"commands" : [
                                    {
                                        u"command_id" : commandName,
                                        u"command_data" : commandData
                                    }   
                                ]
                            }) + u"\n"
    
    try:
        print("In send_erop_command")
        fd, tempPath = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(commandString)
            p = os.popen('/opt/{0}/bin/erop-tunnel -t {1} -s "{2}"'.format(sysname, target, tempPath))
            retVal = p.read()
            p.close()
        finally:
            """"""
            os.remove(tempPath)
    except:
        perror('Unexpected error 1: {0}'.format(sys.exc_info()[0]))
        
    return retVal


def main():
    """MAIN"""
    # Get all properties
    commandData = {
            
        }
    
    # Issue a GetVideoSoureProperties Command
    retVal = send_erop_command("eyerop",
                               "proxycam-ir",
                               "GetVideoSourceProperties",
                               commandData)
    try:
        execResult = json.loads(retVal)[u"proxycam-ir"]
        response = execResult[u"response"]
        errorCode = execResult[u"errorcode"]
        print(u"Error Code: " + errorCode)
        if u"Success" == errorCode:
            videoSourceConfig = response[u"VideoSourceConfiguration"]
            print(u"Active Pipeline: " + response[u"ActivePipeline"])
            print(u"Sensor S/N: " + str(videoSourceConfig[u"Camera Information"][u"SerialNumber"][u"Value"]))
            print(u"BalckHot: " + str(videoSourceConfig[u"Image Processing"][u"BlackHot"][u"Value"]))
            print(u"Histogram Stats: " + videoSourceConfig[u"Camera Information"][u"HistStats"][u"Value"])
    except KeyError as e:
        perror('Key error: {0}'.format(e))
    except ValueError as e:
        perror('Value error: {0}'.format(e))
    except:
        perror('Unexpected error: {0}'.format(sys.exc_info()[0]))
    
    print("Executing command again with request to just Image Processing properties and pretty print the result")
    
    commandData = {
            u"PropertyCategories" : [u"Image Processing"]
        }
    retVal = send_erop_command("eyerop",
                               "proxycam-ir",
                               "GetVideoSourceProperties",
                               commandData)
    try:
        result = json.loads(retVal)
        print(json.dumps(result, indent=4, sort_keys=True))
    except KeyError as e:
        perror('Key error: {0}'.format(e))
    except ValueError as e:
        perror('Value error: {0}'.format(e))
    except:
        perror('Unexpected error: {0}'.format(sys.exc_info()[0]))


if __name__ == '__main__':
    main()