#!/usr/bin/python3
import time
import serial
import struct
import pprint
import math
import UDP_handler


class NavHandler:
    def __init__(self, master):
        dict_keys = ['header', 'gpsTime', 'lon', 'lat', 'alt', 'velNorth', 'velWest', 'velUp', 'posQuality', 'yaw', 'pitch', 'roll', 'yawRate', 'pitchRate', 'rollRate', 'angleError', 'spare', 'checksum']
        unpack_string = '<cLiihhhhhhhhhhhchc'
        expected_header = b'\xff'

        GCS_addr = "10.42.132.37"
        GCS_rcv_local_port = 5002
        socket_sndbuf = 33445566
        socket_rcvbuf = 33445566
        receive_buffer = 1024

        if master: # master is the machine that is connected directly to the navigation box
            self.UART_navReceiver = UART_NavReceiver(dict_keys, unpack_string, expected_header, GCS_addr, GCS_rcv_local_port, socket_sndbuf, socket_rcvbuf)
            self.UART_navReceiver.UART_receiveLoop()
        else: # not a master (slave) machine is a ground station that receives messages from the master
            self.UDP_navReceiver = UDP_NavReceiver(dict_keys, unpack_string, expected_header, GCS_addr, GCS_rcv_local_port, socket_sndbuf, socket_rcvbuf, receive_buffer)
            self.UDP_navReceiver.UDP_receiveLoop()
        
        print("NavHandler closing")
    


# Slave machine (UDP Receiver)
class UDP_NavReceiver:
    def __init__(self, dict_keys, unpack_string, expected_header, GCS_addr, GCS_rcv_local_port, socket_sndbuf, socket_rcvbuf, receive_buffer):
        ## Init parser
        self.navParser = NavParser(dict_keys, unpack_string, expected_header)
        
        ## Init UDP
        self.UDP_receiver = UDP_handler.UDP_receiver(GCS_addr, GCS_rcv_local_port, socket_sndbuf, socket_rcvbuf, receive_buffer)

        self.last_UDP_msg_time = time.time()
    
    def UDP_receiveLoop(self):
        try:
            while True:
                # Receivenessage from UDP
                data = self.UDP_receiver.receive_one()
                
                # Parse message
                self.navParser.parseNavMsg(data)
                
                self.last_UDP_msg_time = printFPS(self.last_UDP_msg_time)

        except KeyboardInterrupt:
            print("Exiting Program")

        except Exception as exception_error:
            print("Error occurred. Exiting Program")
            print("Error: " + str(exception_error))

        finally:
            self.UDP_receiver.close()


# Master machine (UART Receiver, UDP Sender)
class UART_NavReceiver:
    def __init__(self, dict_keys, unpack_string, expected_header, GCS_addr, GCS_rcv_local_port, socket_sndbuf, socket_rcvbuf):
        self.serial_port = serial.Serial(
            port="/dev/ttyTHS1",
            baudrate=19200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_EVEN, #PARITY_EVEN PARITY_MARK PARITY_NONE PARITY_ODD PARITY_SPACE
            stopbits=serial.STOPBITS_ONE, #STOPBITS_ONE STOPBITS_ONE_POINT_FIVE STOPBITS_TWO
        )
        # Wait a second to let the port initialize
        time.sleep(1)
        
        self.pp = pprint.PrettyPrinter(indent=4)

        ## Init Parser
        self.navParser = NavParser(dict_keys, unpack_string, expected_header)

        ## Init UDP
        self.UDP_sender = UDP_handler.UDP_sender(GCS_addr, GCS_rcv_local_port, socket_sndbuf, socket_rcvbuf)

        self.last_UART_msg_time = time.time()

    def UART_receiveLoop(self):
        try:
            while True:
                if self.serial_port.inWaiting() > 0:
                    data = self.serial_port.read(39)

                    # parse the nav message localy
                    self.navParser.parseNavMsg(data)
                    
                    self.last_UART_msg_time = printFPS(self.last_UART_msg_time)

                    # send the nav message to remote station
                    self.UDP_sender.send(data)
                    

        except KeyboardInterrupt:
            print("Exiting Program")

        except Exception as exception_error:
            print("Error occurred. Exiting Program")
            print("Error: " + str(exception_error))

        finally:
            self.serial_port.close()
            pass


class NavParser:
    def __init__(self, dict_keys, unpack_string, expected_header):
        self.dict_keys = dict_keys
        self.unpack_string = unpack_string
        self.expected_header = expected_header

    def parseNavMsg(self, data):
        unpacked = struct.unpack(self.unpack_string, data)
        msg_dict = dict(zip(self.dict_keys, unpacked))
            
        calculated_checksum = (sum([x for x in data]) - data[-1]) % 256
        msg_checksum = int.from_bytes(msg_dict['checksum'], "little")
        
        if msg_dict['header'] != self.expected_header:
            raise Exception("Scheisse!!! Header = {} != {} = Expected header".format(msg_dict['header'], self.expected_header))
        if msg_checksum != calculated_checksum:
            raise Exception("Scheisse!!! msg_checksum = {} != {} = calculated_checksum".format(msg_dict['checksum'], calculated_checksum))
        else:
            # Mesurement unit converstions
            msg_dict['yaw']   = math.degrees(msg_dict['yaw']*math.pi/(2**15))
            msg_dict['pitch'] = math.degrees(msg_dict['pitch']*math.pi/(2**16))
            msg_dict['roll']  = math.degrees(msg_dict['roll']*math.pi/(2**15))

            msg_dict['yawRate']   = math.degrees(msg_dict['yawRate']*math.pi/(2**14))
            msg_dict['pitchRate'] = math.degrees(msg_dict['pitchRate']*math.pi/(2**14))
            msg_dict['rollRate']  = math.degrees(msg_dict['rollRate']*math.pi/(2**14))

            # print message
            print("\
####################################################################################################\n\
Checksum OK, header: {header} gpsTime: {gpsTime} [msec]\n\
(yaw, pitch, roll):      ({yaw:8.2f}, {pitch:8.2f}, {roll:8.2f}) ([-180,180], [-90, 90], [-180, 180]) [deg]\n\
(lat, lon, alt):         ({lat:8}, {lon:8}, {alt:8}) ((2^31)/pi*[rad], (2^31)/pi*[rad], [m])\n\
vel (North, West, Up):   ({velNorth:8}, {velWest:8}, {velUp:8}) [m/sec]\n\
rate (yaw, pitch, roll): ({yawRate:8.2f}, {pitchRate:8.2f}, {rollRate:8.2f}) [rad/sec]\n\
angleError: {angleError} spare: {spare} posQuality: {posQuality} [m] checksum: {checksum}".format(**msg_dict))


def printFPS(last_msg_time):
    now = time.time()
    diff = now - last_msg_time
    try:
        fps = 1/diff
    except ZeroDivisionError:
        fps = 0
    
    print("FPS: {:.2f} Hz".format(fps))

    return now


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Choose if to test a sender or receiver')
    parser.add_argument('--master', action='store_true')
    parser.add_argument('--slave', action='store_true')
    args = parser.parse_args()

    if not args.master and not args.slave:
        print("Please me run with one of the following arguments:")
        print("  --master - if you are running at the master machine (Jetson) which is getting messages from UART")
        print("  --slave  - if you are running at the slave machine (Ground Station) which is getting nav messages from master (Jetson) via UDP")
    elif args.master:
        navHandler = NavHandler(master = True)
    elif args.slave:
        navHandler = NavHandler(master = False)
    else:
        print("Both args --master, --slave were specified. Please specify only one of them")
    
