from scipy import signal, fftpack
import numpy as np
import os
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    # --- Part A - loading saved data (working offline) --- #
    # ----------------------------------------------------- #
    records_path = 'C:\\Users\\ranbenizhak\\OneDrive\\BCI\\BCI_project\\Data\\'
    # records_path = 'C:\\Users\\ranbenizhak\\OneDrive\\BCI\\BCI_project\\Data\\'
    file_names = os.listdir(records_path)
    full_paths = [records_path + fn for fn in file_names if 'Aviv' in fn]
    # currently working with channels 2,5,6,8
    eeg_data = np.loadtxt(full_paths[0], comments='%', skiprows=7, usecols=ACTIVE_CHANNELS, delimiter=',')

    start_diff = int(SAMPLES_PER_FRAME - SAMPLES_PER_FRAME * OVERLAP)
    nWindows = int(eeg_data.shape[0]/start_diff)
    eeg_fft_filtered = []
    eeg_fft_unfilt = []
    for i in range(nWindows):
        eeg_fft_unfilt.append(np.fft.fftshift(fftpack.fft(sub_mean(eeg_data[i*start_diff:i*start_diff+SAMPLES_PER_FRAME, :])), axes=1))
        eeg_data_centered = sub_mean(eeg_data[i*start_diff:i*start_diff+SAMPLES_PER_FRAME, :])  # - np.matlib.repmat(np.mean(eeg_data[i*start_diff:i*start_diff+SAMPLES_PER_FRAME,:], axis=0),SAMPLES_PER_FRAME, 1)
        eeg_fft_filtered.append(np.fft.fftshift(fftpack.fft(butter_bandpass_filter(eeg_data_centered)), axes=1))

    # if PLOT_EXAMPLE:
    #     x_tick = np.arange(0., FS / 2, 1)
    #     r, c = eeg_fft_filtered[0].shape
    #     t = len(eeg_fft_filtered)
    #     plt.figure(2)
    #     ax1 = plt.subplot(211)
    #     averaged_fil = [avg_timeframe(eeg_fft_filtered[k:k + 10])[:int(SAMPLES_PER_FRAME/2), :] for k in np.arange(0, t - 10, 10)]
    #     plt.plot(x_tick, 20 * np.log10(np.abs(averaged_fil[1])))
    #
    #     ax2 = plt.subplot(212)
    #     averaged_unfil = [avg_timeframe(eeg_fft_unfilt[k:k + 10])[:int(SAMPLES_PER_FRAME/2), :] for k in np.arange(0, t - 10, 10)]
    #     plt.plot(x_tick, 20 * np.log10(np.abs(averaged_unfil[1])))

    eeg_list = get_eeg_o_c_b(eeg_fft_filtered, HALF_WIN_AVG)
    configs = 'Overlap factor - ' + str(OVERLAP) + ' || Avg_window - ' + str(HALF_WIN_AVG)
    show_example(eeg_list, configs)




