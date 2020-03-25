# This example demonstrates how to toggle the BlackHot property of the running pipeline
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import json
import getopt
import codecs
from errno import errorcode
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

def send_recieve_command(target, commandName, commandData):
    responseObject = json.loads("{}")
    errorCode = "Success"
    retVal = send_erop_command("eyerop", target, commandName, commandData)
    try:
        execResult = json.loads(retVal)[u"proxycam-ir"]
        responseObject = execResult[u"response"]
        errorCode = execResult[u"errorcode"]
        print("{0} returned {1}".format(commandName, errorCode))
    except KeyError as e:
        perror('Key error: {0}'.format(e))
    except ValueError as e:
        perror('Value error: {0}'.format(e))
    except:
        perror('Unexpected error: {0}'.format(sys.exc_info()[0]))

    return errorCode, responseObject
        

def main():
    """MAIN"""
    # Get current BlackHot value
    commandData = {
            u"PropertyCategories" : [u"Image Processing"],
            u"Image Processing" : {
                u"Properties" : [u"BlackHot"]
                }
        }
    errorCode, response = send_recieve_command("proxycam-ir",
                                               "GetVideoSourceProperties",
                                               commandData)
    try:
        if u"Success" == errorCode:
            videoSourceConfig = response[u"VideoSourceConfiguration"]
            blackHot = videoSourceConfig[u"Image Processing"][u"BlackHot"][u"Value"]
            print ("Current BlackHot Value: {0}".format(blackHot))
            print ("Toggling BlcakHot value")
            
            commandData = {
                    u"VideoSourceConfiguration": {
                        u"Image Processing" :{
                            u"BlackHot" : not blackHot
                            }
                        }
                }
            errorCode, response = send_recieve_command("proxycam-ir",
                                                       "SetVideoSourceProperties",
                                                       commandData)
            try:
                if u"Success" == errorCode:
                    videoSourceConfig = response[u"VideoSourceConfiguration"]
                    print ("New BlackHot Value: {0}".format(videoSourceConfig[u"Image Processing"][u"BlackHot"][u"Value"]))
                else:
                    print ("Unable to set BlcakHot value. Error Code: {0}".format(errorCode))                    
            except KeyError as e:
                perror('SET Key error: {0}'.format(e))
            except ValueError as e:
                perror('SET Value error: {0}'.format(e))
            except:
                perror('SET Unexpected error: {0}'.format(sys.exc_info()[0]))
        else:
            print ("Unable to get BlcakHot value. Error Code: {0}".format(errorCode))                    
                            
    except KeyError as e:
        perror('GET Key error: {0}'.format(e))
    except ValueError as e:
        perror('GET Value error: {0}'.format(e))
    except:
        perror('GET Unexpected error: {0}'.format(sys.exc_info()[0]))


if __name__ == '__main__':
    main()