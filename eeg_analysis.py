from scipy import signal, fftpack
import numpy as np
import os
import matplotlib.pyplot as plt

ACTIVE_CHANNELS = (2, 5, 6, 8)
FS = 256
LOW_CUT = 2.5
HIGH_CUT = 45
ORDER = 5

def datestr2num(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + float(m) * 60 + float(s)

##Function: butter_bandpass
##Description: returns the buttersworth coefficient for the band pass filter
##Parameters: lowcut, highcut - cutoff frequencies, fs- sample rate, order of filter
##Return: a,b - coefficients
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    print(str(high), '  ', str(low))
    b, a = signal.butter(order, [low, high], btype='band')
    #b, a = signal.butter(order, low)
    return b, a


def fir_bandpass(lowcut=LOW_CUT, highcut=HIGH_CUT, fs=FS, order=ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    h = signal.firwin(order, [low, high], pass_zero=False)
    return h

##Function: butter_bandpass_filter
##Description: filters the date by the produced buttersworth band pass filter
##Parameters: data, lowcut, highcut - cutoff frequencies, fs- sample rate, order of filter
##Return: filtered data
def butter_bandpass_filter(data, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=FS, order=ORDER, plotPlease=False):
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
    assert eeg_list[0].shape == (256, len(ACTIVE_CHANNELS))
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


# --- Part A - loading saved data (working offline) --- #
# ----------------------------------------------------- #
records_path = 'C:\\Users\\ranbenizhak\\OneDrive\\BCI\\BCI_project\\Data\\'
file_names = os.listdir(records_path)
full_paths = [records_path + fn for fn in file_names if 'Aviv' in fn]
# currently for debugging, running with first item
# currently working with channels 2,5,6,8
eeg_data = np.loadtxt(full_paths[0], comments='%', skiprows=6, usecols=ACTIVE_CHANNELS, delimiter=',')

new_shape = (int(eeg_data.shape[0] / FS) , FS, eeg_data.shape[1])
# cutting the data into 1 second frames - cutting the samples number to multiplication of f_sample
eeg_data_per_sec = np.reshape(eeg_data[0: new_shape[0]*FS, :], new_shape)
assert np.array_equal(eeg_data_per_sec[0, :, :] , eeg_data[0:FS, :]) # sanity check
t, r, c = eeg_data_per_sec.shape
eeg_data_centered = [sub_mean(eeg_data_per_sec[x, ...]) for x in range(t)]

# --- Part B - filtering the data ------------------- #
# --------------------------------------------------- #
# BPF - passing  2.5 - 45 [Hz]
eeg_filtered = [butter_bandpass_filter(eeg_data_centered[x]) for x in range(t)]

# # Visualizing results
# eeg_example_unfilt = fftpack.fft(eeg_data_per_sec[10, :, 0])
# eeg_example_filt = fftpack.fft(eeg_filtered[10][:, 0])
x_tick = np.arange(0., FS/2, 0.5)
# fig = plt.figure(2)
# plt.plot(x_tick, 20*np.log10(np.abs(eeg_example_unfilt)), 'r--', x_tick, 20*np.log10(np.abs(eeg_example_filt)), 'b--')
# plt.show()

# --- Part C - averaging time frames and plotting

# average every 10 time frames
averaged = [avg_timeframe(eeg_filtered[k:k+10]) for k in np.arange(0, t-10, 10)]
plt.plot(x_tick, 20*np.log10(np.abs(averaged[1])))



