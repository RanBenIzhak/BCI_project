from scipy import signal
import numpy as np
import os
import matplotlib.pyplot as plt

def datestr2num(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + float(m) * 60 + float(s)


records_path = 'C:\\Users\\ranbenizhak\\OneDrive\\BCI\\OpenBCI_application.windows64\\SavedData\\'
file_names = os.listdir(records_path)
full_path = records_path + file_names[-1]

eeg_data = np.loadtxt(full_path, comments='%', skiprows=6, usecols=(2,3,4,5,6,7,8), delimiter=',')
f_sample = 250
Pxx_den = []
print(eeg_data.shape)
#     specto[i] = scipy.signal.spectogram(eeg_data[:,i])
for i in range(7):
    print(eeg_data[:, i])
    f, Pxx_den = signal.welch(eeg_data[:, i], f_sample)
    plt.semilogy(f,Pxx_den)
    plt.ylim([0.5e-3, 10])
    plt.xlabel('frequency[Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
