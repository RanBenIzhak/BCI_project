from scipy import signal, fftpack
import numpy as np
from numpy import linalg as LA
import os
import matplotlib.pyplot as plt
import diffusion_maps as dm
import math
from mpl_toolkits.mplot3d import Axes3D
import time


ACTIVE_CHANNELS = (2, 5, 6, 8)
FS = 250
LOW_CUT = 2.5
HIGH_CUT = 40
ORDER = 8       # up to 8
OVERLAP=0.5     #   0-0.5   ##############
TIME_FRAME= 1      #[Sec]        ##############
SAMPLES_PER_FRAME= TIME_FRAME * FS      #(TIME_FRAME/FS)*SAMPLES_PER_FRAME<=1 ##############
HALF_WIN_AVG = 5
PLOT_EXAMPLE = False

def datestr2num(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + float(m) * 60 + float(s)

def butter_bandpass(lowcut, highcut, fs, order):
    '''
    ##Function: butter_bandpass
    ##Description: returns the buttersworth coefficient for the band pass filter
    ##Parameters: lowcut, highcut - cutoff frequencies, fs- sample rate, order of filter
    ##Return: a,b - coefficients
    :param lowcut: 
    :param highcut: 
    :param fs: 
    :param order: 
    :return: 
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # print(str(high), '  ', str(low))
    b, a = signal.butter(order, [low, high], btype='band')
    #b, a = signal.butter(order, low)
    return b, a

def fir_bandpass(lowcut=LOW_CUT, highcut=HIGH_CUT, fs=FS, order=ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    h = signal.firwin(order, [low, high], pass_zero=False)
    return h

def butter_bandpass_filter(data, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=FS, order=ORDER, plotPlease=False):
    '''
    ##Function: butter_bandpass_filter
    ##Description: filters the date by the produced buttersworth band pass filter
    ##Parameters: data, lowcut, highcut - cutoff frequencies, fs- sample rate, order of filter
    ##Return: filtered data
    :param data: 
    :param lowcut: 
    :param highcut: 
    :param fs: 
    :param order: 
    :param plotPlease: 
    :return: 
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if plotPlease:
        x_tick = np.arange(0., FS / 2, 0.25)
        w, h = signal.freqz(b, a)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label='buterworth order = %d' % order)
        plt.show(block=False)
    y = signal.lfilter(b, a, data, axis=0)
    return y

def avg_timeframe(eeg_list):
    '''   
    :param eeg_list: list of eeg recordings, each in shape [samples, channels]
    :return: average signal in frequency domain for the given eeg list
    '''
    assert eeg_list[0].shape == (SAMPLES_PER_FRAME, len(ACTIVE_CHANNELS))
    eeg_sum = np.zeros_like(eeg_list[0])
    for i in range(len(eeg_list)):
        eeg_sum += abs(fftpack.fft(eeg_list[i], axis=0))
    return eeg_sum / len(eeg_list)

def sub_mean(eeg_frame):
    '''
    substract the mean of each channel (each column) from each channel
    :param eeg_frame: 2d array of [samples, channels]
    :return: [ch1 - mean(ch1), ch2 - mean(ch2), ... ]
    '''
    rows = eeg_frame.shape[0]
    eeg_frame_mean = np.matlib.repmat(np.mean(eeg_frame, axis=0), rows, 1)
    return eeg_frame - eeg_frame_mean
# -------- Functions end --------------

def show_example(eeg_ex_list, config_string):
    '''
    Visualizing results of experiment
    :param eeg_ex_list: 
    :param config_string: 
    :return: 
    '''
    fig, ax = plt.subplots(nrows=3, ncols=3)
    plt.suptitle('Frequency analysis of eeg for Open, Closed, Blink states')
    fig.text(0.5, 0.9, config_string, ha='center')
    fig.text(0.5, 0.04, 'F [Hz]', ha='center')
    fig.text(0.07, 0.5, '|Amp| [dB]', va='center', rotation='vertical')
    fig.text(0.04, 0.23, 'Blink', va='center', rotation='vertical')
    fig.text(0.04, 0.5, 'Closed', va='center', rotation='vertical')
    fig.text(0.04, 0.77, 'Open', va='center', rotation='vertical')
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(20* np.log10(np.abs(eeg_ex_list[i][j][:int(SAMPLES_PER_FRAME/4), :])))
    return


def get_eeg_o_c_b(eeg_in, half_avg_win):
    '''
    Slicing the eeg signal to Open, Closed, Blink, and averaging, for each, 
    the surrounding timeframes
    ** CONSTANTS ARE ACCORDING TO SPECIFIC EXPERIMENT SETTINGS **
    :param eeg_in: eeg input signal
    :param half_avg_win: distance (in frames) to average signal on 
    :return: list of [eeg_in[open], eeg_in[closed], eeg_in[blinking] ] 
    '''
    frames_to_second = int(1 / (TIME_FRAME * (1 - OVERLAP)))
    # set according to desired specific experiment and your settings, constants in [seconds]
    time_frames = {'open': range(10 * frames_to_second, 240 * frames_to_second, 40 * frames_to_second),
                   'blink': range(30 * frames_to_second, 240 * frames_to_second, 80 * frames_to_second),
                   'closed': range(70 * frames_to_second, 240 * frames_to_second, 80 * frames_to_second)}

    eeg_open = [avg_timeframe(eeg_in[x - half_avg_win:x + half_avg_win]) for x in time_frames['open']]
    eeg_closed = [avg_timeframe(eeg_in[x - half_avg_win:x + half_avg_win]) for x in time_frames['closed']]
    eeg_blink = [avg_timeframe(eeg_in[x - half_avg_win:x + half_avg_win]) for x in time_frames['blink']]
    return [eeg_open, eeg_closed, eeg_blink]

def get_aviv_exp_timeframes(eeg_in):
    '''
    ** According to experiment specific parameters **
    extracting labels and legend for Aviv experiment
    :param eeg_in: 
    :return: labels, legend
    '''
    legend = {0: 'open',
              1: 'blink',
              2: 'closed'}
    session_len = len(eeg_in)
    samples_per_mode = int(20 / (TIME_FRAME * OVERLAP) )
    labels = np.zeros(session_len)
    start_indices = range(samples_per_mode, session_len, 2 * samples_per_mode)

    for j, start_ind in enumerate(start_indices):
        if start_ind+20 >= session_len:
            continue
        if j % 2 == 0:
            # asserting blink label
            labels[start_ind:start_ind+20] = 1
        else:
            # asserting closed
            labels[start_ind:start_ind + 20] = 2
    return labels, legend

def load_and_filter_data(path, filter=True):
    '''
    loads the file in the specific path,
    dividing it to windows according to global parameters,
    and performing FFT + Filtering of the signal
    :param path: 
    :return: FFT unfiltered
             FFT filtered
    '''
    eeg_data = np.loadtxt(path, comments='%', skiprows=7, usecols=ACTIVE_CHANNELS, delimiter=',')
    start_diff = int(SAMPLES_PER_FRAME - SAMPLES_PER_FRAME * OVERLAP)
    nWindows = int(eeg_data.shape[0] / start_diff)
    eeg_fft_filtered = []
    eeg_fft_unfilt = []
    for i in range(nWindows):
        if filter:
            eeg_fft_unfilt.append(
                np.fft.fftshift(fftpack.fft(sub_mean(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME, :])),
                                axes=1))
            eeg_data_centered = sub_mean(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME,
                                         :])  # - np.matlib.repmat(np.mean(eeg_data[i*start_diff:i*start_diff+SAMPLES_PER_FRAME,:], axis=0),SAMPLES_PER_FRAME, 1)
            eeg_fft_filtered.append(np.fft.fftshift(fftpack.fft(butter_bandpass_filter(eeg_data_centered)), axes=1))
        else:
            eeg_fft_unfilt.append(sub_mean(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME, :]))

    return eeg_fft_unfilt, eeg_fft_filtered


def show_diffusion(coordinates, labels_list, legend):
    colors = ['red', 'green', 'blue']
    i = 0
    for coords, labels in zip(coordinates, labels_list):
        fig = plt.figure()
        a = np.asarray(coords)
        x = a[:, 0]
        if a.shape[1] > 1:
            y = a[:, 1]
        else:
            y = np.zeros(x.shape)
        if a.shape[1] == 3:
            ax = fig.gca(projection='3d')
            z = a[:, 2]
            for label in legend:
                cur_label = legend[label]
                Axes3D.scatter(ax, np.asarray(x[labels==label]), np.asarray(y[labels==label]),
                               np.asarray(z[labels==label]), c=colors[label], label=cur_label)
                plt.show()

        else:
            ax = fig.gca()
            for label in legend:
                cur_label = legend[label]
                ax.scatter(np.asarray(x[labels==label]), np.asarray(y[labels==label]),
                           c=colors[label], label=cur_label)
                plt.show()
        i += 1

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # --- Part A - loading saved data (working offline) --- #
    # ----------------------------------------------------- #
    cur_path = os.path.dirname(os.path.realpath(__file__))
    records_path = os.path.join(cur_path, 'Data')
    # records_path = 'C:\\Users\\ranbenizhak\\OneDrive\\BCI\\BCI_project\\Data\\'
    file_names = os.listdir(records_path)
    full_paths = [os.path.join(records_path, fn) for fn in file_names if 'Aviv' in fn]

    # # ==== Displaying averaging examples for different modes (open/closed/blink) ==== #
    # # =================  currently working with channels 2,5,6,8 ==================== #

    # eeg_fft_unfilt, eeg_fft_filtered = load_and_filter_data(full_paths[0])

    # eeg_list = get_eeg_o_c_b(eeg_fft_filtered, HALF_WIN_AVG)
    # configs = 'Overlap factor - ' + str(OVERLAP) + ' || Avg_window - ' + str(HALF_WIN_AVG)
    # show_example(eeg_list, configs)

    # ======= Diffusion maps for data ========= #
    coords_out, labels_out = [], []
    for data_path in full_paths:   # for each experiment
        eeg_fft_unfilt, eeg_fft_filt = load_and_filter_data(data_path)
        # eeg_unfilt, eeg_filt = load_and_filter_data(data_path, filter=False)
        labels, legend = get_aviv_exp_timeframes(eeg_fft_unfilt)
        eeg_flatten = np.asarray([x.flatten() for x in eeg_fft_unfilt])

        epsilon = 1000  # diffusion distance epsilon
        coords, dataList = dm.diffusionMapping(np.transpose(eeg_flatten[:-1]),
                                                lambda x, y: math.exp(-LA.norm(x - y) / epsilon),
                                                t=2, dim=3)
        labels_out.append(labels[:-1])
        coords_out.append(coords)
    show_diffusion(coords_out, labels_out, legend)
    ttt = 5






