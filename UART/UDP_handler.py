import socket
import time


class UDP_sender:
    
    def __init__(self, send_to_addr, send_to_port, socket_sndbuf, socket_rcvbuf):
        self.send_to_addr = send_to_addr
        self.send_to_port = send_to_port
        self.socket_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_send.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, socket_sndbuf)
        self.socket_send.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, socket_rcvbuf)
        print("-"*20)
        print("Sender initialized")
        
    def send_loop(self):
        msg_count = 0
        while msg_count < 20:
            msg = "msg(" + str(msg_count) + ")"
            self.send(msg)
            msg_count += 1
            time.sleep(0.5)

    def send(self, msg):
        #print("Sending: {}".format(msg))
        try:
            if isinstance(msg, bytes):
                self.socket_send.sendto(msg, (self.send_to_addr, self.send_to_port))
            elif isinstance(msg, str):
                self.socket_send.sendto(msg.encode('utf-8'), (self.send_to_addr, self.send_to_port))
        except Exception as e:
            print("Failed to send data to UDP channel. Error: {}".format(e))
        
    def close(self):
        self.socket_send.close()

class UDP_receiver:
    def __init__(self, recv_addr, recv_port, socket_sndbuf, socket_rcvbuf, receive_buffer):
        self.recv_addr = recv_addr
        self.recv_port = recv_port
        self.receive_buffer = receive_buffer
        self.socket_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_recv.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, socket_sndbuf)
        self.socket_recv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, socket_rcvbuf)
        self.socket_recv.bind((self.recv_addr, recv_port))
        print("-"*20)
        print("Receiver initialized")

    def receive_loop(self):
        while True:
            data = self.socket_recv.recv(self.receive_buffer)
            if not data:
                break
            print(data)
    
    def receive_one(self):
        return self.socket_recv.recv(self.receive_buffer)
    
    def close(self):
        self.socket_recv.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Choose if to test a sender or receiver')
    parser.add_argument('--send_test', action='store_true')
    parser.add_argument('--recv_test', action='store_true')
    args = parser.parse_args()

    A_ADDR = "127.0.0.1"
    A_SEND_LOCAL_PORT = 6001
    A_RECV_LOCAL_PORT = 6002
    
    B_ADDR = "127.0.0.1"
    B_SEND_LOCAL_PORT = 5001
    B_RECV_LOCAL_PORT = 5002
    
    # Sending example
    if args.send_test:
        print("Sender test")
        A_sender = UDP_sender(B_ADDR, B_RECV_LOCAL_PORT, 33445566, 33445566)
        msg = "msg to send"
        A_sender.send(msg)
        A_sender.send_loop()
        A_sender.close()
    
    elif args.recv_test:
        print("Receiver test")
        B_receiver = UDP_receiver(B_ADDR, B_RECV_LOCAL_PORT, 33445566, 33445566, 1024)
        B_receiver.receive_loop()
        B_receiver.close()

    # TODO threading example
    # TODO duplex connection