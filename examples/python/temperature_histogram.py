# coding=utf-8
from __future__ import print_function, division, unicode_literals
import numpy as np
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time

import sys
sys.path.append(os.path.dirname(__file__))
from facesIDtracker import facesIdTracker


class cyclicBuffer(object):
    """
    Implement cyclic buffer using ndarrays.
    """
    def __init__(self, shape):
        """
        Initialize cyclicBuffer instance.

        Parameters
        ----------
        shape : tuple
            Buffer shape, where shape[0] is number of elements in the buffer, each of which with shape shape[1:]

        Returns
        ----------
            None
        """
        # initialize buffer
        self.buf = np.zeros(shape)

        # initialize indices
        self.indRead = np.array(0) # index from which to start reading
        self.indWrite = np.array(0) # index to which start writing

        # save buffer length
        self.length =np.array(shape[0]) # buffer length


    def write(self, x):
        """
        Write x to buffer

        Parameters
        ----------
        x : ndarray
            Array to write to buffer

        Returns
        ----------
        None.

        """
        # get x length
        N = x.shape[0]

        #calculate write indices
        inds = (self.indWrite + np.arange(N)) % self.length

        # write new data to buffer
        self.buf[inds] = x

        # update write index
        self.indWrite = (self.indWrite + N) % self.length


    def read(self, N, advance_read_index=False):
        """
        Read N elements from buffer.

        Parameters
        ----------
        N : ndarray
            Number of elements to be read from buffer.

        Returns
        ----------
        y : ndarray
            Data read from buffer

        """

        # calculate read indices
        inds = (self.indRead + np.arange(N)) % self.length

        # read data from buffer
        y = self.buf[inds]

        if advance_read_index:
            # update read index
            self.indRead = (self.indRead + N) % self.length

        return y

    def getNumOfElementsToRead(self):

        """
        Calculate how much un-read elements can be read.

        Parameters
        ----------
        None

        Returns
        ----------
        N : ndarray
            Number of un-read elements in buffer.
        """

        N = self.indWrite - self.indRead

        if N < 0:
            N += self.length

        return N

    def num_elements_written(self):

        N = self.indRead - self.indWrite

        if N < 0:
            N += self.length

        return N


class TemperatureHistogram(object):


    def __init__(self,
                 hist_calc_interval=30 * 60, # [sec]
                 buffer_max_len=120*60*1,  # [minutes * sec * persons_per_sec]
                 hist_percentile=0.85,  # [%]
                 N_samples_for_temp_th=50,
                 N_samples_for_first_temp_th=20,
                 temp_th_nominal=34.0,
                 temp_th_min=30.0,
                 temp_th_max=36.0,
                 ):

        self.hist_calc_interval = hist_calc_interval  # [sec]
        self.buffer_max_len = buffer_max_len  # [sec]
        self.hist_percentile = hist_percentile   # [%]
        self.N_samples_for_temp_th = N_samples_for_temp_th
        self.N_samples_for_first_temp_th = N_samples_for_first_temp_th
        self.temp_th_nominal = temp_th_nominal
        self.temp_th_min = temp_th_min
        self.temp_th_max = temp_th_max
        self.is_initialized = False  # True after first temp_th calculation

        # Initialize Cyclic Buffer
        self.shape_element = (3,)  #  each buffer element is comprised of ndarray of [time, temp, id]
        self.shape_buffer = (self.buffer_max_len,) + self.shape_element
        self.buffer = cyclicBuffer(self.shape_buffer)

        # initialize temperature threshold
        self.temp_th = self.temp_th_nominal

        # initialize face tracker
        self.faces_tracker = facesIdTracker()


    def write_sample(self, time_stamp, temp, id=[None]):

        time_stamp = np.array(time_stamp)
        temp = np.array(temp)
        id = np.array(id)

        element = np.stack((time_stamp, temp, id), axis=1)

        self.buffer.write(element)

    def num_elements_written(self):

        return self.time_buffer.num_elements_written()


    def num_elements_in_time_interval(self, time_current, time_interval):

        # read all time
        time_vec_all = self.buffer.read(self.buffer.length)[:, 0]

        # find indices of wanted time interval
        time_th = time_current - time_interval
        ind = np.where(time_vec_all > time_th)[0]

        # get temperature values of wanted time interval
        time_vec = time_vec_all[ind]

        N = len(time_vec)

        return N


    def read_elements_in_time_interval(self, time_current, time_interval):

        # read all time
        data = self.buffer.read(self.buffer.length)
        time_vec_all = data[:, 0]
        temp_vec_all = data[:, 1]
        id_vec_all = data[:, 2]

        # find indices of wanted time interval
        time_th = time_current - time_interval
        ind = np.where(time_vec_all > time_th)[0]

        # get data values of wanted time interval
        time_vec = time_vec_all[ind]
        temp_vec = temp_vec_all[ind]
        id_vec = id_vec_all[ind]
        N = len(ind)

        return time_vec, temp_vec, id_vec, N


    def read_N_elements(self, N):

        data = self.buffer.read(N)
        time_vec = data[:, 0]
        temp_vec = data[:, 1]
        id_vec = data[:, 2]
        N = len(time_vec)

        return time_vec, temp_vec, id_vec, N


    def calculate_temperature_threshold(self, time_current, hist_calc_interval=None, display=False):

        if hist_calc_interval is None:
            hist_calc_interval = self.hist_calc_interval

        bins, hist, temp_percentage, N_samples = \
            TemperatureHistogram.calculate_temperature_histogram(self.buffer, hist_calc_interval, time_current,
                                                                 hist_percentile=self.hist_percentile, display=display)

        alpha = float(N_samples) / self.N_samples_for_temp_th
        alpha = np.clip(alpha, 0., 1.)

        temp_th = alpha * temp_percentage + (1 - alpha) * self.temp_th_nominal

        temp_th = np.clip(temp_th, a_min=self.temp_th_min, a_max=self.temp_th_max)

        temp_th = np.round(temp_th, 1)

        self.temp_th = temp_th

        if not self.is_initialized:
            self.is_initialized = True

        return temp_th


    @staticmethod
    def calculate_temperature_histogram(buffer, hist_calc_interval, time_current, hist_percentile=0.85, display=False):

        # read all data from buffers
        data = buffer.read(buffer.length)
        time_vec_all = data[:, 0]
        temp_vec_all = data[:, 1]
        id_vec_all = data[:, 2]

        # find indices of wanted time interval
        time_th = time_current - hist_calc_interval
        ind = np.where((time_vec_all > time_th) & (temp_vec_all > 0))[0]

        # get temperature values of wanted time interval
        temp_vec = temp_vec_all[ind]

        # calculate histogram
        bin_edges = np.arange(25.0, 40.0, 0.1)
        bins, hist, y_percentage = TemperatureHistogram.calc_hist(x=temp_vec, bin_edges=bin_edges,
                                                                  cdf_percentage=hist_percentile, display=display)

        N_samples = len(temp_vec)

        return bins, hist, y_percentage, N_samples


    @staticmethod
    def calc_hist(x, bin_edges=np.arange(25.0, 40.0, 0.1), cdf_percentage=0.8, display=True):

        # clip values
        x = np.clip(x, a_min=np.min(bin_edges), a_max=np.max(bin_edges))

        # calculate histogram
        hist, bin_edges = np.histogram(x, bins=bin_edges, range=(x.min(), x.max()), density=True)

        bins_offset = 0.5 * (bin_edges[1] - bin_edges[0])
        bins = bin_edges[1:] - bins_offset

        bins_diff = np.min(np.min(bins[1:] - bins[:-1]))

        # normalize histogram (calc pdf)
        hist = hist / hist.sum()

        # calculate cdf
        cdf = np.cumsum(hist)

        # find cdf percentage
        f = interpolate.interp1d(x=cdf, y=bins, kind='linear', axis=-1, copy=True, bounds_error=None,
                                 fill_value=(cdf[0], cdf[-1]), assume_sorted=False)

        y_percentage = f(cdf_percentage)

        if display:

            fontsize = 20.

            hf = plt.figure()

            # plot histogram
            leg_str = '$\mu={:.1f}$  |  $\sigma={:.2}$'.format(x.mean(), x.std())
            plt.bar(bins, hist, width=0.8 * bins_diff, alpha=0.6, color='g', label=leg_str)

            # plot cdf
            plt.plot(bins, cdf, label='cdf')

            # plot percentage
            leg_str = 'Temp. @ {:.0f}% CDF = {:.1f}'.format(cdf_percentage * 100, y_percentage)
            p = plt.plot(y_percentage, cdf_percentage, label=leg_str, marker='o', markersize=12, markerfacecolor='None')
            plt.plot([y_percentage, y_percentage, 0], [0, cdf_percentage, cdf_percentage], linestyle=':',
                     color=p[0].get_color(), markerfacecolor='None')

            plt.legend(prop={'size': fontsize})

            title_str = 'Temprature Histogram - {} Examples'.format(len(x))
            plt.title(title_str, fontsize=fontsize)

            plt.ylabel('Prevalence [%]', fontsize=fontsize)
            plt.xlabel('Temperature [Celsius]')

            # make image more pretty
            axes = plt.gca()
            x_lim = [x.min(), x.max()]
            if x_lim[0] == x_lim[1]:
                x_lim[1] += 1.
            axes.set_xlim(x_lim)

            plt.grid()

            plt.tight_layout()
            pyplot_maximize_plot()
            plt.show(block=False)
            plt.pause(1e-3)

        return bins, hist, y_percentage


    def write_faces_temperature(self, faces):

        """
        faces : list
            Each element in faces is a tuple of (box, temperature)
            where box is ndarray of [left, top, right, bottom]
        """

        # track faces ids

        # update faces temprature

def pyplot_maximize_plot():

    plot_backend = matplotlib.get_backend()
    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend == 'Qt4Agg':
        mng.window.showMaximized()


def read_and_display_image(imgpath, filename, display=False):

    if filename.endswith(".png"):
        print(os.path.join(imgpath, filename))
        fig = plt.figure()
        image = mpimg.imread(os.path.join(imgpath, filename))
        if display:
            plt.title(filename)
            plt.imshow(image)
            plt.show()
            plt.pause(0.1)
    return image, int(filename[0:-4])

def detect_faces(image, image_id):
    #On real life Run RetinaDet
    #this is just my debug code:
    if (image_id%3) == 0 :
        #current_frame_faces is composed of [(np.array([TL_X, TL_Y, BR_X, BR_y]), Temp)]
        current_frame_faces = [(np.array([10,20,30,40]), 37.0), (np.array([200, 100, 300, 400]), 38), (np.array([50, 50, 150, 150]), 36)]
    elif (image_id%3) == 1 :
        current_frame_faces = [(np.array([15,20,30,40]),37.5),  (np.array([200,100,300,400])-500, 38.) , (np.array([50,50,150,150])+20, 16.5)]
    else:
        current_frame_faces = [(np.array(20,20,30,40), 38)]

    return current_frame_faces

def write_element_to_temperature_histogram(current_frame_faces, temperature_histogram):
    # get current time
    time_stamp = np.array([time.time()])
    # write temp and time_stamp to buffer
    reshaped_current_frame_faces = current_frame_faces.reshape(current_frame_faces.shape[0]*current_frame_faces.shape[1],1)
    temperature_histogram.write_sample(temp=reshaped_current_frame_faces, time_stamp=time_stamp)
    return current_frame_faces

def pop_last_faces_from_histogram(temperature_histogram):
    last_written_index = temperature_histogram.buffer.indWrite
    last_read_index = temperature_histogram.buffer.indRead
    last_frame_faces = temperature_histogram.buffer.buf[(last_read_index):(last_written_index)]
    reshaped_last_frame_faces = last_frame_faces.reshape(-1,6)
    return reshaped_last_frame_faces

def unq_searchsorted(A,B):

    # Get unique elements of A and B and the indices based on the uniqueness
    unqA,idx1 = np.unique(A,return_inverse=True)
    unqB,idx2 = np.unique(B,return_inverse=True)

    # Create mask equivalent to np.in1d(A,B) and np.in1d(B,A) for unique elements
    mask1 = (np.searchsorted(unqB,unqA,'right') - np.searchsorted(unqB,unqA,'left'))==1
    mask2 = (np.searchsorted(unqA,unqB,'right') - np.searchsorted(unqA,unqB,'left'))==1

    # Map back to all non-unique indices to get equivalent of np.in1d(A,B),
    # np.in1d(B,A) results for non-unique elements
    return mask1[idx1],mask2[idx2]

def check_for_new_faces_and_update_buffers(image, image_id, current_frame_faces, temperature_histogram, temp_th_nominal):

    #_to_buffer lists consist of [bbox , temp, id]
    last_frame_faces_from_buffer = pop_last_faces_from_histogram(temperature_histogram)
    frame_faces_to_buffer =[]
    new_frame_faces = []
    #check for new faces

    #if no faces detected in current frame - exit from function:
    if len(current_frame_faces) == 0:
        return []

    #if there were no faces in last frame write faces as is to buffer
    if len(last_frame_faces_from_buffer) == 0:
        frame_faces_to_buffer = current_frame_faces
        ids, meanTemp = temperature_histogram.faces_tracker.giveFacesIds(current_frame_faces[:, 0:4], current_frame_faces[:, 4])
        if len(ids) != len(current_frame_faces[:,0]):
            return []
        frame_faces_to_buffer = np.append(current_frame_faces, ids.reshape(ids.shape[0],1), axis=1)
        write_element_to_temperature_histogram(frame_faces_to_buffer, temperature_histogram)
        return frame_faces_to_buffer
    #if there were faces in last frame check if any of them are new
    else:
        lastFrameFaces = last_frame_faces_from_buffer[:, 0:4]
        lastFrameTemp  = last_frame_faces_from_buffer[:, 4]
        lastFrameIds   = last_frame_faces_from_buffer[:, 5]
        currFrameFaces = current_frame_faces[:, 0:4]
        currFrameTemp  = current_frame_faces[:, 4]
        currFrameIds, meanTemp2 = temperature_histogram.faces_tracker.giveFacesIds(currFrameFaces, currFrameTemp)
        ids_of_new_faces_in_frame = currFrameIds - lastFrameIds
        #maskCurr, maskLast = unq_searchsorted(currFrameIds,lastFrameIds)
        for i,val in enumerate(ids_of_new_faces_in_frame):
            if (val>0): #only for new ids
                new_frame_faces_line = np.append(np.append(currFrameFaces[i], currFrameTemp[i]), currFrameIds[i].astype(int))
                if len(new_frame_faces) == 0: #first line deal seperatly
                    new_frame_faces = new_frame_faces_line
                else:
                    new_frame_faces = np.vstack([new_frame_faces, new_frame_faces_line])

        print(image_id, '- ids_of_new_faces_in_frame: ', ids_of_new_faces_in_frame)
        if not all([v == 0 for v in ids_of_new_faces_in_frame]):
            write_element_to_temperature_histogram(new_frame_faces, temperature_histogram)
        return new_frame_faces


def show_boxes(im, two_d_current_frame_faces, scale = 1.0):
    plt.cla()
    plt.axis("off")
    plt.imshow(im)

    for det in two_d_current_frame_faces :
        bbox = det[:4] * scale
        color = (1, 1, 1)
        rect = plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=color, linewidth=2.5)
        plt.gca().add_patch(rect)

        plt.gca().text(bbox[0], bbox[1],
                       '{:s}'.format(str(det[4])),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')

        plt.gca().text(bbox[0]+20, bbox[1],
                       'ID = {:s}'.format(str(det[5])),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='red')

    plt.show()
    return im

def display_new_people_on_image(image, image_id, current_frame_faces):
    global displayImage
    two_d_current_frame_faces=[]

    if len(current_frame_faces):
        two_d_current_frame_faces = current_frame_faces.reshape(-1,6)

    if (displayImage == True):
        show_boxes(image, two_d_current_frame_faces, scale=1.0)


def main_simple_temperature_histogram():
    # ------------------------
    # simulation parameters
    # ------------------------

    # simulate people flow
    expected_number_of_people_hour = 100.
    lambda_poisson_hour = expected_number_of_people_hour / 60.  #  60 minutes
    lambda_poisson = lambda_poisson_hour / (60.)  # probability per second

    # temperature
    tmp_mean = 33.5  # [celsius]
    temp_std = 1.  # [celsius]

    # # simulation length
    N_sec = 2 * 60 * 60  # [sec]

    display = True

    # ------------------------
    # temperature histogram parameters
    # ------------------------

    hist_calc_interval = 30 * 60  # [sec]
    hist_percentile = 0.85
    N_samples_for_temp_th = 50
    N_samples_for_first_temp_th = 20
    temp_th_nominal = 34.0
    buffer_max_len = 3000  #
    temp_th_min = 30.0
    temp_th_max = 36.0

    temp_hist = TemperatureHistogram(hist_calc_interval=hist_calc_interval,
                                     hist_percentile = hist_percentile,
                                     N_samples_for_temp_th=N_samples_for_temp_th,
                                     N_samples_for_first_temp_th=N_samples_for_first_temp_th,
                                     temp_th_nominal=temp_th_nominal,
                                     buffer_max_len=buffer_max_len,
                                     temp_th_min=temp_th_min,
                                     temp_th_max=temp_th_max,
                                     )

    # time_start = time.time()
    time_start = 0

    for n in range(N_sec):

        # time_current = time.time()
        time_current = n

        # sample people enterance
        prob_people = np.random.poisson(lam=lambda_poisson, size=(1,))[0]

        if prob_people > 0.5:

            # sample temprature
            temp = np.array([np.random.normal(tmp_mean, temp_std)])

            # get current time
            time_stamp = np.array([time_current])

            # write temp and time_stamp to buffer
            temp_hist.write_sample(temp=temp, time_stamp=time_stamp)

        # initalize temp_th
        if not temp_hist.is_initialized and (temp_hist.buffer.getNumOfElementsToRead() > temp_hist.N_samples_for_first_temp_th):
            temp_th = temp_hist.calculate_temperature_threshold(time_current=time_current, hist_calc_interval=temp_hist.hist_calc_interval, display=display)

        # calculate temperature histogram
        if (np.mod(n, temp_hist.hist_calc_interval) == 0) and (n > 0) and (temp_hist.buffer.getNumOfElementsToRead() > temp_hist.N_samples_for_first_temp_th):
            temp_th = temp_hist.calculate_temperature_threshold(time_current=time_current, hist_calc_interval=temp_hist.hist_calc_interval, display=display)


    print ('Done')


def main_temperature_histogram_with_face_tracking():

    # time_buffer_max_len = 30 * 60 * 7  # [num_minutes * sec_in_minute * images_per_sec]
    # temp_th_nominal = 34.0
    # people_buffer_max_len = 3000
    # temp_th_min = 30.0
    # temp_th_max = 36.0
    #
    # # ------------------------------
    # # Images folder to simulate on:
    # # ------------------------------
    #
    # # ------------------------------------
    # # Temperature histogram initialization
    # # ------------------------------------
    # temperature_histogram = TemperatureHistogram(time_buffer_max_len=time_buffer_max_len,
    #                                              hist_percentile=0.85,
    #                                              num_people_for_temp_th=50,
    #                                              num_people_for_first_temp_th=20,
    #                                              temp_th_nominal=temp_th_nominal,
    #                                              people_buffer_max_len=people_buffer_max_len,
    #                                              temp_th_min=temp_th_min,
    #                                              temp_th_max=temp_th_max,
    #                                              )

    imgpath = '/home/moshes2/Projects/face_fever_detection/data/2020_30_03__11_09_03/png_im/'
    display = True

    hist_calc_interval = 30 * 60  # [sec]
    hist_percentile = 0.85
    N_samples_for_temp_th = 50
    N_samples_for_first_temp_th = 20
    temp_th_nominal = 34.0
    buffer_max_len = 3000  #
    temp_th_min = 30.0
    temp_th_max = 36.0

    temp_hist = TemperatureHistogram(hist_calc_interval=hist_calc_interval,
                                     hist_percentile = hist_percentile,
                                     N_samples_for_temp_th=N_samples_for_temp_th,
                                     N_samples_for_first_temp_th=N_samples_for_first_temp_th,
                                     temp_th_nominal=temp_th_nominal,
                                     buffer_max_len=buffer_max_len,
                                     temp_th_min=temp_th_min,
                                     temp_th_max=temp_th_max,
                                     )

    facesTracker = facesIdTracker()

    # ------------------------------
    #           Main loop:
    # ------------------------------
    for filename in sorted(os.listdir(imgpath)):

        image, image_id = read_and_display_image(imgpath, filename, display)

        current_frame_faces = detect_faces(image, image_id)

        new_faces_from_buffer = check_for_new_faces_and_update_buffers(image, image_id, current_frame_faces, temp_hist, temp_th_nominal)

        display_new_people_on_image(image, image_id, new_faces_from_buffer)




if __name__ == '__main__':

    # main_simple_temperature_histogram()

    main_temperature_histogram_with_face_tracking()

    print ('Done')


