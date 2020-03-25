import Jetson.GPIO as GPIO
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pin', type=int, default=4)
args = parser.parse_args()
print("Got arg: {}".format(args.pin))



class GPIO_listener:
    def __init__(self, input_pin):
        self.last_1pps = time.time()
        self.input_pin = input_pin
        GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
        GPIO.setup(self.input_pin, GPIO.IN) # set pin as an Input pin
        GPIO.add_event_detect(self.input_pin, GPIO.RISING, callback=self.callback_raisingEdge)
        print("Start listening! CTRL+C to stop")
        self.diff_list = []

    def callback_raisingEdge(self, channel):
        now = time.time()
        diff = now - self.last_1pps
        self.diff_list.append(diff)
        print("1PPS diff = {}".format(diff))
        self.last_1pps = now

    def main_loop(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            GPIO.remove_event_detect(self.input_pin)
            print("Pin {} event removed".format(self.input_pin))
            self.statistics()
        finally:
            GPIO.cleanup()

    def statistics(self):
        try:
            self.diff_list.pop(0)# remove 1st diff, in order not to harm statistics
        except IndexError:
            print("Diff list is empty. Nothing to build a statistics with:(")
            return
        print("Statistics collected. Importing matplotlib")
        import matplotlib.pyplot as plt
        import numpy as np
        std = np.std(self.diff_list, dtype=np.float64)
        avg = np.average(self.diff_list)
        
        plt.plot(self.diff_list, 'bo')
        plt.plot(self.diff_list, 'k')
        plt.ylabel('1PPS time period')
        plt.xlabel('avg {:.4f} std {:.4f}'.format(avg, std))
        print("Calculating statistics, plotting the graph ...")
        plt.show()
        try:
            import pdb; pdb.set_trace()
            self.diff_list # you can manipulate the statistics list now if you want to. Ctrl+C to exit
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    _GPIO_listener = GPIO_listener(args.pin)
    _GPIO_listener.main_loop()

