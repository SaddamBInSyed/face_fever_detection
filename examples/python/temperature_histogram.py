import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time

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


    def read(self, N):
        """
        Read N elements from buffer.

        Parameters
        ----------
        N : ndarray
            Number of elements to be readfrom buffer.

        Returns
        ----------
        y : ndarray
            Data read from buffer

        """

        # calculate read indices
        inds = (self.indRead + np.arange(N)) % self.length

        # read data from buffer
        y = self.buf[inds]

        # update read index
        self.indRead = (self.indRead + N) % self.length

        return y

    def getNumOfElementsToRead(self):

        """
        Calculate how much un-readelements can be read.

        Parameters
        ----------
        None

        Returns
        ----------
        N : ndarray
            Number of un-read elements inbuffer.
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
                 hist_calc_interval=30*60,  # [sec]
                 hist_percentile=0.85,  # [%]
                 N_samples_for_temp_th=50,
                 N_samples_for_first_temp_th=20,
                 temp_th_nominal=34.0,
                 buffer_max_len=3000,
                 temp_th_min=30.0,
                 temp_th_max=36.0,
                 ):

        self.hist_calc_interval = hist_calc_interval  # [sec]
        self.hist_percentile = hist_percentile   # [%]
        self.N_samples_for_temp_th = N_samples_for_temp_th
        self.temp_th_nominal = temp_th_nominal
        self.temp_th_min = temp_th_min
        self.temp_th_max = temp_th_max

        self.is_initialized = False  # True after first temp_th calculation

        # cyclic buffer
        self.buffer_max_len = buffer_max_len
        # -----------------

        self.shape_element = (1,)
        self.shape_buffer = (self.buffer_max_len,) + self.shape_element


        self.temp_buffer = cyclicBuffer(self.shape_buffer)
        self.time_buffer = cyclicBuffer(self.shape_buffer)

        self.temp_th = self.temp_th_nominal


    def write_sample(self, temp, time_stamp):

        temp = np.array(temp)
        time_stamp = np.array(time_stamp)

        self.temp_buffer.write(temp)
        self.time_buffer.write(time_stamp)

    def num_elements_written(self):

        return self.time_buffer.num_elements_written()


    def num_elements_in_time_interval(self, time_interval):

        # read all time
        time_vec_all = self.time_buffer.read(self.time_buffer.length)

        # find indices of wanted time interval
        time_th = time_current - time_interval
        ind = np.where(time_vec_all > time_th)[0]

        # get temperature values of wanted time interval
        time_vec = time_vec_all[ind]

        N = len(time_vec)

        return N



    def calculate_temperature_threshold(self, time_current, hist_calc_interval=None, display=False):

        if hist_calc_interval is None:
            hist_calc_interval = self.hist_calc_interval

        bins, hist, temp_percentage, N_samples = \
            TemperatureHistogram.calculate_temp_hist(self.time_buffer, self.temp_buffer, hist_calc_interval, time_current, hist_percentile=hist_percentile, display=display)

        alpha = float(N_samples) / N_samples_for_temp_th
        alpha = np.clip(alpha, 0., 1.)

        temp_th = alpha * temp_percentage + (1 - alpha) * temp_th_nominal

        temp_th = np.clip(temp_th, a_min=self.temp_th_min, a_max=self.temp_th_max)

        temp_th = np.round(temp_th, 1)

        self.temp_th = temp_th

        if not self.is_initialized:
            self.is_initialized = True

        return temp_th


    @staticmethod
    def calculate_temp_hist(time_buffer, temp_buffer, hist_calc_interval, time_current, hist_percentile=0.8,
                            display=False):

        # read all data from buffers
        temp_vec_all = temp_buffer.read(temp_buffer.length)
        time_vec_all = time_buffer.read(time_buffer.length)

        # find indices of wanted time interval
        time_th = time_current - hist_calc_interval
        ind = np.where((time_vec_all > time_th) & (temp_vec_all > 0))[0]

        # get temperature values of wanted time interval
        temp_vec = temp_vec_all[ind]

        # calculate histogram
        bin_edges = np.arange(25.0, 40.0, 0.1)
        bins, hist, y_percentage = TemperatureHistogram.calc_hist(x=temp_vec, bin_edges=bin_edges, cdf_percentage=hist_percentile,
                                             display=display)

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
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show(block=False)
            plt.pause(1e-3)

        return bins, hist, y_percentage


if __name__ == '__main__':

    # ------------------------
    # simulation parameters
    # ------------------------

    # simulate people flow
    expected_number_of_people_hour = 100
    lambda_poisson_hour = expected_number_of_people_hour / 60  #  60 minutes
    lambda_poisson = lambda_poisson_hour / (60)  # probability per second

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
        if not temp_hist.is_initialized and (temp_hist.num_elements_in_time_interval(time_current - time_start) > N_samples_for_first_temp_th):
            temp_th = temp_hist.calculate_temperature_threshold(time_current=time_current, hist_calc_interval=temp_hist.hist_calc_interval, display=display)

        # calculate temprature histogram
        if (np.mod(n, temp_hist.hist_calc_interval) == 0) and (n > 0):
            temp_th = temp_hist.calculate_temperature_threshold(time_current=time_current, hist_calc_interval=temp_hist.hist_calc_interval, display=display)


    print ('Done')


