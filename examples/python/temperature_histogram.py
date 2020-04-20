# coding=utf-8
from __future__ import print_function, division, unicode_literals
import numpy as np
from scipy import interpolate
from scipy.special import erf
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time

import sys
sys.path.append(os.path.dirname(__file__))
from facesIDtracker import facesIdTracker
from findDC import CFindDC


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


    def rewrite(self, x):
        """
        Rewrite x to buffer, by replacing last len(x) elements by x.

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

        # rewind indWrite
        self.indWrite = (self.indWrite - N) % self.length

        # write x
        self.write(x)


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


    def read_last_elements(self, N):
            """
            Read last N elements from buffer.

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
            indReadStart = self.indWrite - N
            inds = (indReadStart + np.arange(N)) % self.length

            # read data from buffer
            y = self.buf[inds]

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

        self.hist_calc_every_N_sec = 15 * 60  # [sec]
        self.hist_calc_interval = hist_calc_interval  # [sec]
        self.buffer_max_len = buffer_max_len  # [sec]
        self.hist_percentile = hist_percentile   # [%]
        self.N_samples_for_temp_th = N_samples_for_temp_th
        self.min_N_samples_for_temp_th = 10
        self.N_samples_for_first_temp_th = N_samples_for_first_temp_th
        self.temp_th_nominal = temp_th_nominal
        self.temp_th_min = temp_th_min
        self.temp_th_max = temp_th_max
        self.is_initialized = False  # True after first temp_th calculation
        self.start_time = time.time()

        # Initialize Cyclic Buffer
        self.shape_element = (3,)  #  each buffer element is comprised of ndarray of [time, temp, id]
        self.shape_buffer = (self.buffer_max_len,) + self.shape_element
        self.buffer = cyclicBuffer(self.shape_buffer)

        # initialize temperature threshold
        self.temp_th = self.temp_th_nominal

        # initialize face tracker
        self.faces_tracker = facesIdTracker()

        # initialize find temperture DC offset
        self.find_DC = CFindDC()
        self.dc_offset = - 2.5  # default core-forehead temperature offset
        self.temp_th_when_using_dc_offset = 37.5

        #
        self.use_temperature_histogram = True
        self.use_temperature_statistics = True

        # temperature statistics
        self.prior_mu_sigma = (36.77, 0.6)
        self.sigma_measure_given_temp = 0.4

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
        time_vec_all = self.buffer.read_last_elements(self.buffer.length)[:, 0]

        # find indices of wanted time interval
        time_th = time_current - time_interval
        ind = np.where(time_vec_all > time_th)[0]

        # get temperature values of wanted time interval
        time_vec = time_vec_all[ind]

        N = len(time_vec)

        return N


    def read_elements_in_time_interval(self, time_current, time_interval):

        # read all buffer
        data = self.buffer.read_last_elements(self.buffer.length)
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

        data = self.buffer.read_last_elements(N)
        time_vec = data[:, 0]
        temp_vec = data[:, 1]
        id_vec = data[:, 2]
        N = len(time_vec)

        return time_vec, temp_vec, id_vec, N

    def calculate_temp_statistic(self, time_current, hist_calc_interval=None):

        time_vec_all, temp_vec_all, id_vec_all, N = self.read_N_elements(self.buffer.length)

        # find indices of wanted time interval
        time_th = time_current - hist_calc_interval
        ind = np.where((time_vec_all > time_th) & (temp_vec_all > 0))[0]

        # get temperature values of wanted time interval
        temp_vec = temp_vec_all[ind]

        # calc DC offset using temp measurements
        dcMaxLike, dcMeanLike = self.find_DC.findDC(temp_vec)
        offset = dcMaxLike

        # calculating measurements statistic
        measure_mean = np.mean(temp_vec)
        measure_std = np.std(temp_vec)
        measure_mu_sigma = (measure_mean, measure_std)

        return offset, measure_mu_sigma

    def estimate_temp(self, curr_temp_measure, offset, measure_mu_sigma):

        mu_prior = self.prior_mu_sigma[0]
        sigma_prior = self.prior_mu_sigma[1]
        mu_measure = curr_temp_measure - offset
        sigma_measure_given_temp = self.sigma_measure_given_temp

        # calculating mean and std of joined distribution (the mean is the estimated temperature)
        joined_mu = (mu_prior * sigma_measure_given_temp ** 2 + mu_measure * sigma_prior ** 2) / (sigma_prior ** 2 + sigma_measure_given_temp ** 2)
        joined_sigma = np.sqrt(sigma_prior ** 2 * sigma_measure_given_temp ** 2 / (sigma_prior ** 2 + sigma_measure_given_temp ** 2))

        temp_estimation = joined_mu

        return temp_estimation

    def write_elements(self, time_vec, temp_vec, id_vec):

        data = np.stack((time_vec, temp_vec, id_vec), axis=1)

        self.buffer.write(data)


    def rewrite_elements(self, time_vec, temp_vec, id_vec):

        data = np.stack((time_vec, temp_vec, id_vec), axis=1)

        self.buffer.rewrite(data)



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
        data = buffer.read_last_elements(buffer.length)
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


    def find_id_indices(self, id_vec, ids_to_search, N=100):

        # find indices of ids in buffer
        indices_in_buffer = np.where(np.isin(id_vec, ids_to_search, assume_unique=True))[0]

        # find indices of found ids in ids_to_search
        indices_in_ids_to_search = np.where(np.isin(ids_to_search, id_vec, assume_unique=True))[0]

        return indices_in_buffer, indices_in_ids_to_search



    def update_faces_temperature(self, faces_list, temp_list, time_stamp, temp_memory=0.4, N=100):

        """
        Updated faces temperature in buffer.
        Faces ids' are tracked using facesIdTracker():
            - for existing faces, temperature and time stamp are updated.
            - for new faces, values are added to buffer.

        faces_list : list
            List of faces bounding boxes, each element is ndarray of [left, top, right, bottom]
        temp_list : list
            List of temperatures corresponding to faces in faces_list.
        time_stamp : int
            Current time in seconds.
        temp_memory: float, optional
            Weight of previous temperature value.
        N : int, optional
            Number of last buffer elements in which faces will be searched.
        """

        id_faces, indices_faces_existing_ids = None, None

        if len(faces_list) > 0:

            # track faces ids
            faces_array = np.stack(faces_list, axis=0)
            id_faces, temp_mean = self.faces_tracker.giveFacesIds(faces_array, temp_list)

            # -------------------------
            # update faces temperature
            # -------------------------

            # read last N elements from buffer
            if N == -1:
                N = self.buffer.length

            # read data from buffer
            time_vec, temp_vec, id_vec, N = self.read_N_elements(N)

            # search for faces_id in buffer
            indices_buffer_existing_ids, indices_faces_existing_ids = self.find_id_indices(id_vec, ids_to_search=id_faces)

            # update temperatures and time of existing ids
            for ind_buffer, ind_faces in zip(indices_buffer_existing_ids, indices_faces_existing_ids):

                # get current temperature
                temp_prev = temp_vec[ind_buffer]

                # get new temperature
                temp_new = temp_list[ind_faces]

                # calculate updated temperature and time
                time_updated = time_stamp
                temp_updated = temp_memory * temp_prev + (1 - temp_memory) * temp_new

                # update arrays
                time_vec[ind_buffer] = time_updated
                temp_vec[ind_buffer] = temp_updated

            # rewrite updated arrays
            self.rewrite_elements(time_vec, temp_vec, id_vec)

            # write new faces data
            inds_new = np.setdiff1d(np.arange(len(faces_list)), indices_faces_existing_ids)
            time_vec_new = np.ones_like(inds_new) * time_stamp
            temp_vec_new = np.array(temp_list)[inds_new]
            id_vec_new = id_faces[inds_new]

            self.write_elements(time_vec_new, temp_vec_new, id_vec_new)

        return id_faces, indices_faces_existing_ids


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
            plt.show(block=False)
            plt.pause(0.1)
    return image, int(filename[0:-4])

def detect_faces(image, image_id):
    #On real life Run RetinaDet
    #this is just my debug code:
    if (image_id%3) == 0 :
        #current_frame_faces is composed of [(np.array([TL_X, TL_Y, BR_X, BR_y]), Temp)]
        current_frame_faces = [np.array([10,20,30,40]), np.array([200, 100, 300, 400]), np.array([50, 50, 150, 150])]
        current_temps = [10.0, 20, 30]
    elif (image_id%3) == 1 :
        current_frame_faces = [np.array([15,20,30,40]),  np.array([200,100,300,400])-80, np.array([50,50,150,150])+20]
        current_temps = [100.0, 200, 300]
    else:
        current_frame_faces = [np.array([20,20,30,40])]
        current_temps = [400]
    return current_frame_faces, current_temps

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


def show_boxes(im, two_d_current_frame_faces, temp_list, id_faces, existing_faces_flags, scale = 1.0):
    plt.cla()
    plt.axis("off")
    plt.imshow(im)

    color_new = (0, 0, 1)
    color_exist = (1, 0, 0)

    for det, temp, id, is_exist in zip(two_d_current_frame_faces, temp_list, id_faces, existing_faces_flags) :

        bbox = det[:4] * scale

        color = color_exist if is_exist else color_new

        rect = plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=color, linewidth=2.5)
        plt.gca().add_patch(rect)

        plt.gca().text(bbox[0], bbox[1],
                       '{:s}'.format(str(temp)),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')

        plt.gca().text(bbox[0]+20, bbox[1],
                       'ID = {:s}'.format(str(id)),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='red')

    plt.show()
    return im


def display_new_people_on_image(image, image_id, current_frame_faces, temp_list, id_faces, indices_faces_existing_ids, display=True):

    #
    existing_faces_flags = np.isin(np.arange(len(current_frame_faces)), indices_faces_existing_ids)

    if display:
        show_boxes(image, current_frame_faces, temp_list, id_faces, existing_faces_flags, scale=1.0)


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

            temp_estimation = temp_hist.estimate_temp(temp, temp_hist.dc_offset, measure_mu_sigma=None)

        # initalize temp_th
        if not temp_hist.is_initialized and (temp_hist.buffer.getNumOfElementsToRead() > temp_hist.N_samples_for_first_temp_th):
            temp_th = temp_hist.calculate_temperature_threshold(time_current=time_current, hist_calc_interval=temp_hist.hist_calc_interval, display=display)
            temp_hist.dc_offset, measure_mu_sigma = temp_hist.calculate_temp_statistic(temp, time_current=time_current, hist_calc_interval=temp_hist.hist_calc_interval)
            temp_estimation = temp_hist.estimate_temp(temp, temp_hist.dc_offset, measure_mu_sigma=None)

        # calculate temperature histogram
        if (np.mod(n, temp_hist.hist_calc_interval) == 0) and (n > 0) and (temp_hist.buffer.getNumOfElementsToRead() > temp_hist.N_samples_for_first_temp_th):
            temp_th = temp_hist.calculate_temperature_threshold(time_current=time_current, hist_calc_interval=temp_hist.hist_calc_interval, display=display)
            temp_hist.dc_offset, measure_mu_sigma = temp_hist.calculate_temp_statistic(temp, time_current=time_current, hist_calc_interval=temp_hist.hist_calc_interval)
            temp_estimation = temp_hist.estimate_temp(temp, temp_hist.dc_offset, measure_mu_sigma=None)


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

    imgpath = 'data/2020_30_03__11_09_03/png_im/'
    display = True

    hist_calc_interval = 30 * 60  # [sec]
    hist_percentile = 0.85
    N_samples_for_temp_th = 50
    N_samples_for_first_temp_th = 20
    temp_th_nominal = 34.0
    buffer_max_len = 3000  #
    temp_th_min = 30.0
    temp_th_max = 36.0

    # faces tracker temperature
    temp_memory = 0.25

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
    for n, filename in enumerate(sorted(os.listdir(imgpath))):

        # time_current = time.time()
        time_current = n

        image, image_id = read_and_display_image(imgpath, filename, display)

        current_frame_faces, current_temps = detect_faces(image, image_id)

        # new_faces_from_buffer = check_for_new_faces_and_update_buffers(image, image_id, current_frame_faces, temp_hist, temp_th_nominal)

        id_faces, indices_faces_existing_ids = temp_hist.update_faces_temperature(faces_list=current_frame_faces, temp_list=current_temps, time_stamp=time_current, temp_memory=temp_memory)

        display_new_people_on_image(image, image_id, current_frame_faces, current_temps, id_faces, indices_faces_existing_ids)




if __name__ == '__main__':

    main_simple_temperature_histogram()

    # main_temperature_histogram_with_face_tracking()

    print ('Done')


