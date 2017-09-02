from scipy import signal, fftpack
import numpy as np
import os
import matplotlib.pyplot as plt

ACTIVE_CHANNELS = (2, 5, 6, 8)
FS = 256
LOW_CUT = 2.5
HIGH_CUT = 45
ORDER = 6


def datestr2num(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + float(m) * 60 + float(s)

##Function: butter_bandpass
##Description: returns the buttersworth coefficient for the band pass filter
##Parameters: lowcut, highcut - cutoff frequencies, fs- sample rate, order of filter
##Return: a,b - coefficients
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a


##Function: butter_bandpass_filter
##Description: filters the date by the produced buttersworth band pass filter
##Parameters: data, lowcut, highcut - cutoff frequencies, fs- sample rate, order of filter
##Return: filtered data
def butter_bandpass_filter(data, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=FS, order=ORDER, plotPlease=False):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if plotPlease:
        w, h  = signal.freqz(a, b)
        h_dB = 20 * np.log10(np.abs(h))
        fig = plt.figure(10)
        plt.plot(w/np.max(w), h_dB)
        plt.grid()
        plt.show()
    y = signal.lfilter(b, a, data)
    return y

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

# --- Part B - filtering the data --- #

# BPF - passing  2.5 - 45 [Hz]
eeg_filtered = butter_bandpass_filter(eeg_data_per_sec, plotPlease=True)
eeg_example_unfilt = fftpack.fft(eeg_data_per_sec[0, :, 0])
eeg_example_filt = fftpack.fft(eeg_filtered[0, :, 0])
fig = plt.figure(2)
plt.plot(np.real(eeg_example_unfilt))
plt.show()

fig2 = plt.figure(3)
plt.plot(np.real(eeg_example_filt))
plt.show()



