# This example demonstrates how to switch the running pipeline using erop-tunnel
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

def usage():
    print ("Usage:\npython switchpipeline PIPELINE")
    
def parse_args(argv):
    try:        
        opts, args = getopt.getopt(argv[1:], "", [])
    except getopt.GetoptError as err:
        # print help information and exit:
        print (str(err))
        usage()
        sys.exit(2)
    
    if len(args) != 1:
        usage()
        sys.exit(2)
    
    return args[0]


def main():
    """MAIN"""
    
    newPipeline = parse_args(sys.argv).decode('unicode-escape') # We use this so user can switch to "OPGAL EYE-Q™" using string "OPGAL EYE-Q\u2122" from command line
    
    # Get active pipeline
    commandData = {
            
        }
    
    # Issue a EnumVideoPipelines Command
    erroCode, response = send_recieve_command("proxycam-ir",
                                              "EnumVideoPipelines",
                                              commandData)
    try:
        print(u"Active Pipeline: " + response[u"ActivePipeline"])
        supportedPipelines = response[u"SupportedPipelines"]
        if newPipeline in supportedPipelines:
            print(u"Switching pipeline to " + newPipeline)
            commandData = {
                u"PipelineId" : newPipeline
                }
            errorCode, response = send_recieve_command("proxycam-ir", 
                                                      "SetVideoPipeline", 
                                                      commandData)
            if "Success" == errorCode:
                commandData = {
                        "RebuildPipeline": True
                    }
                errorCode, response = send_recieve_command("proxycam-ir", 
                                                          "StartVideoPipeline", 
                                                          commandData)
                if "Success" == errorCode:
                    print(u"Active Pipeline: " + response[u"ActivePipeline"])
                    if response[u"PipelineRunning"]:
                        print("Pipeline is Running")
                    else:
                        print("Pipeline is Stopped")
                else:
                    print("Unable to start new video pipeline")
            else:                
                print(u"SetVideoPipeline failed with error: {0}".format(response[u"ErrorMessage"]))
        else:
            print("{0} is not in the list of supported pipelines".format(newPipeline))
    except KeyError as e:
        perror('Key error: {0}'.format(e))
    except ValueError as e:
        perror('Value error: {0}'.format(e))
    except:
        perror('Unexpected error: {0}'.format(sys.exc_info()[0]))
    
    

if __name__ == '__main__':
    main()