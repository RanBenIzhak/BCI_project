from scipy import signal, fftpack
import numpy as np
from numpy import linalg as LA
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import diffusion_maps as dm
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import cross_val_score
from sklearn import manifold, decomposition, svm
import pickle
from data_handler import write_xml

# ===== Filter settings ===== #
ORDER = 2       # up to 8
LOW_CUT = 2.5
HIGH_CUT = 22
FFT_VECTOR_CUT_LOW=5
FFT_VECTOR_CUT_HIGH=50

# ===== Diffusion parameters ===== #
DIM_RNG_LOW = 1
DIM_RNG_HIGH = 11
EPSILON_LOW = 100
EPSILON_HIGH = 100000

# ===== KNN constants ===== #
N_NEIGHBORS_LOW = 3
N_NEIGHBORS_HIGH = 9
n_components = 5 #ALL
n_neighbors = 5  #LLE, Isomap

# ===== Time frame settings ===== #
FS = 255
OVERLAP = 0.5     #   [0,1)
TIME_FRAME = 1.6 #2      #[Sec]        ##############
SAMPLES_PER_FRAME = int(TIME_FRAME * FS)      #(TIME_FRAME/FS)*SAMPLES_PER_FRAME<=1 ##############

# ===== Run Configurations ===== #
EXPERIMENT_TYPE = 2  # 1 - old ran\aviv's experiment, 2 - new (6 participants) experiment, 3 - Igor only results
FREQUENCY_ANALYSIS_METHOD = 'welch'      #'fft'/'welch'
if EXPERIMENT_TYPE==1:
    ACTIVE_CHANNELS = (2, 5, 6, 8)
elif EXPERIMENT_TYPE==2:
    ACTIVE_CHANNELS = (2, 3, 4, 6, 7)
    IGOR_CHANNELS = (1, 3, 4)
elif EXPERIMENT_TYPE ==3:
    ACTIVE_CHANNELS = (1, 3, 4)

# ===== Setting for experiment 1 only ===== #
## Do not change unless you want three states and that's for Experiment 1 data only
SKIP_BLINK = 1
if SKIP_BLINK == 0:
    SVM = 0
else:
    SVM = 1


# ===== General configs ===== #
SAVE_FIGS = 1                                                               # Flag 1- save figures after plotting them to file, 0- don't save figures
HALF_WIN_AVG = 3                                                            # Variable for setting frame average window size for 5 frames average set to 3
EXAMPLES_VIZUAL_WINDOW = 15                                                 # variable for setting frequency, time and average time frames we are looking at

# ===== Welch settings ===== #
WELCH_SEGMENTS = 4
OVERLAP_WELCH = 0.5
WELCH_TIME_FRAME = (TIME_FRAME*(1+(WELCH_SEGMENTS-1)*(1-OVERLAP_WELCH)))
if FREQUENCY_ANALYSIS_METHOD=='welch':
    TIME_FRAME=WELCH_TIME_FRAME
    SAMPLES_PER_FRAME = int(TIME_FRAME * FS)

if FREQUENCY_ANALYSIS_METHOD == 'fft':
    TIME_FRAME_PARTIAL = TIME_FRAME
else :
    TIME_FRAME_PARTIAL = WELCH_TIME_FRAME

# ==== Some constants for global use ==== #
N_epsilon_parameter = int(math.log10(EPSILON_HIGH / EPSILON_LOW))  # requires running new DM
epsilon_array = np.asarray([EPSILON_LOW * math.pow(10, i) for i in range(int(N_epsilon_parameter))])
knn_neibours_array = np.asarray(range(N_NEIGHBORS_LOW, N_NEIGHBORS_HIGH, 2))
N_neibours_parameter = len(knn_neibours_array)  # does not require neq
N_experiments = 0
N_dim_parameter = DIM_RNG_HIGH - DIM_RNG_LOW

# ==== String constants ===== #
UNFILTERED_STRING = 'Unfiltered'
FILTERED_STRING = 'Filtered'
WELCH_STRING = 'Welch'
FFT_STRING = 'FFT'
if FREQUENCY_ANALYSIS_METHOD=='fft':
    ANALYSIS_METHOD_STRING = FFT_STRING
elif FREQUENCY_ANALYSIS_METHOD=='welch':
    ANALYSIS_METHOD_STRING = WELCH_STRING

# ==== PATHS AND FILE NAMES ==== #
cur_path = os.path.dirname(os.path.realpath(__file__))
RECORDS_PATH = os.path.join(cur_path, 'Data', 'Experiment ' + str(EXPERIMENT_TYPE))
PATH_RESULTS_DIR = os.path.join(cur_path, 'Results')
CUR_TRIAT_PATH=''
SEPRATE_ANALYSIS_OUTPUT_FILE = 'Seprate_analysis_output.txt'
MEANS_ANALYSIS_OUTPUT_FILE = 'Means_analysis_output.txt'
CONFIG_ANALYSIS_OUTPUT_FILE = 'Config.txt'
FILTER_ANALYSIS_OUTPUT_FILE = 'Filter.png'
SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ = 'Signal_examples_freq_'+ANALYSIS_METHOD_STRING+'_'
SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ_AVG = 'Signal_examples_freq_avg_'+FFT_STRING+'_'
SIGNAL_EXAMPLE_OUTPUT_FILE_TIME = 'Signal_examples_time_'
DIFFUSION_EXAMPLE_OUTPUT_FILE = 'Diffusion_'+ANALYSIS_METHOD_STRING+'_'



#########################
# ===== Functions ===== #
#########################

# ===== Util ===== #

def datestr2num(time_str):
    '''
    datestr2num - Util
    Description : Return the number of seconds that have passed at a given time point past 00:00:00
    :param time_str:  Dates string
    :return: Number of seconds after 00:00:00
    '''
    h, m, s = time_str.split(':')
    return int(h) * 3600 + float(m) * 60 + float(s)

def get_triat_name(exp_ind,path):
    '''
    get_triat_name - Util
    Description:
    :param exp_ind:
    :param path:
    :return:
    '''
    path =path.strip()
    res=path.split('-')
    return res[0].strip()

def avg_frame(eeg_list):
    '''
    avg_frame - Util
    Description: calculates data average of frames list
    :param eeg_list: list of eeg recordings, each in shape [samples, channels]
    :return: average signal in frequency domain for the given eeg list
    '''
    assert eeg_list[0].shape == (SAMPLES_PER_FRAME, len(ACTIVE_CHANNELS))
    eeg_avg = np.zeros_like(eeg_list[0])
    for i in range(len(eeg_list)):
        eeg_avg += eeg_list[i]/ len(eeg_list)
    return eeg_avg

def avg_timeframe(eeg_list):
    '''
    avg_timeframe - Util
    Description: same as the previous function avg_frame but without the assertion
    :param eeg_list: list of eeg recordings, each in shape [samples, channels]
    :return: average signal in frequency domain for the given eeg list
    '''
    #assert eeg_list[0].shape == (SAMPLES_PER_FRAME, len(ACTIVE_CHANNELS))
    eeg_sum = np.zeros_like(eeg_list[0])
    for i in range(len(eeg_list)):
        eeg_sum += abs(fftpack.fftshift(fftpack.fft(eeg_list[i], axis=0),axes=0))
    return eeg_sum / len(eeg_list)

def sub_mean(eeg_frame):
    '''
    sub_mean - Util
    substract the mean of each channel (each column) from each channel
    :param eeg_frame: 2d array of [samples, channels]
    :return: [ch1 - mean(ch1), ch2 - mean(ch2), ... ]
    '''
    rows = eeg_frame.shape[0]
    eeg_frame_mean = np.matlib.repmat(np.mean(eeg_frame, axis=0), rows, 1)
    return eeg_frame - eeg_frame_mean

def mode_time_frame():
    '''
    mode_time_frame - Util
    Description - returns the time frame of each mode according to the experiment configuration
    :return: Time frame for each mode in seconds
    '''
    if EXPERIMENT_TYPE==1:
        return 20
    elif EXPERIMENT_TYPE==2:
        return 150
    elif EXPERIMENT_TYPE==3:
        return 150

def chunkify(lst, n):
    '''
    chunkify - Util
    :param lst:
    :param n:
    :return:
    '''
    return [lst[i::n] for i in range(n)]

def extract_state(data, labels, label_to_extract):
    '''
    extract_state - Util
    Description: Extract certain data with certain labels - for ignoring blinking data
    :param data: data farmes list
    :param labels: data frames labels list
    :param label_to_extract: labels number to extract
    :return: frames and labels of extracted labels
    '''
    cur_d = [d1 for d1, l1 in zip(data, labels) if l1 == label_to_extract]
    cur_l = [l1 for l1 in labels if l1 == label_to_extract]
    return cur_d, cur_l

def calcMeans(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt):
    '''
    calcMeans - Utils
    Description : This function calculates the means of the results over all experiments.

    :param exp_mean_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_std_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_eigvals_unfilt[exp_ind][dim_ind][epsilon_ind]:
    :param svm_mean_unfilt[exp_ind]:
    :param svm_std_unfilt[exp_ind]:
    :param exp_mean_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_std_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_eigvals_filt[exp_ind][dim_ind][epsilon_ind]:
    :param svm_mean_filt[exp_ind]:
    :param svm_std_filt[exp_ind]:
    :return:    mean_exp_mean_filt[dim_ind][epsilon_ind][KNNeibours_ind],
                mean_exp_std_filt[dim_ind][epsilon_ind][KNNeibours_ind],
                mean_exp_eigvals_filt[dim_ind][epsilon_ind],
                mean_svm_mean_filt,
                mean_svm_std_filt,
                mean_exp_mean_unfilt[dim_ind][epsilon_ind][KNNeibours_ind],
                mean_exp_std_unfilt[dim_ind][epsilon_ind][KNNeibours_ind],
                mean_exp_eigvals_unfilt[dim_ind][epsilon_ind],
                mean_svm_mean_unfilt,
                mean_svm_std_unfilt
    '''
    mean_exp_mean_unfilt, mean_exp_std_unfilt, mean_exp_eigvals_unfilt = [[[[0.0] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)], [[[[0.0] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)], [[[] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)]
    mean_exp_mean_filt, mean_exp_std_filt, mean_exp_eigvals_filt = [[[[0.0] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)], [[[[0.0] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)], [[[] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)]
    mean_svm_mean_unfilt = 0.0
    mean_svm_std_unfilt = 0.0
    mean_svm_mean_filt = 0.0
    mean_svm_std_filt = 0.0
    for i in range(N_experiments):
        mean_svm_mean_filt = mean_svm_mean_filt + svm_mean_filt[i] / N_experiments
        mean_svm_std_filt = mean_svm_std_filt + svm_std_filt[i] / N_experiments
        mean_svm_mean_unfilt = mean_svm_mean_unfilt + svm_mean_unfilt[i] / N_experiments
        mean_svm_std_unfilt = mean_svm_std_unfilt + svm_std_unfilt[i] / N_experiments
        for j in range(N_dim_parameter):
            for k in range(N_epsilon_parameter):
                if (i == 0):
                    mean_exp_eigvals_filt[j][k] = exp_eigvals_filt[i][j][k] / N_experiments
                    mean_exp_eigvals_unfilt[j][k] = exp_eigvals_unfilt[i][j][k] / N_experiments
                else:
                    mean_exp_eigvals_filt[j][k] = mean_exp_eigvals_filt[j][k] + exp_eigvals_filt[i][j][k] / N_experiments
                    mean_exp_eigvals_unfilt[j][k] = mean_exp_eigvals_unfilt[j][k] + exp_eigvals_unfilt[i][j][k] / N_experiments

                for l in range(N_neibours_parameter):
                    mean_exp_mean_filt[j][k][l] = mean_exp_mean_filt[j][k][l] + exp_mean_filt[i][j][k][l] / N_experiments
                    mean_exp_std_filt[j][k][l] = mean_exp_std_filt[j][k][l] + exp_std_filt[i][j][k][l] / N_experiments
                    mean_exp_mean_unfilt[j][k][l] = mean_exp_mean_unfilt[j][k][l] + exp_mean_unfilt[i][j][k][l] / N_experiments
                    mean_exp_std_unfilt[j][k][l] = mean_exp_std_unfilt[j][k][l] + exp_std_unfilt[i][j][k][l] / N_experiments
    return mean_exp_mean_filt, mean_exp_std_filt, mean_exp_eigvals_filt, mean_svm_mean_filt, mean_svm_std_filt,  \
        mean_exp_mean_unfilt,mean_exp_std_unfilt, mean_exp_eigvals_unfilt , mean_svm_mean_unfilt, mean_svm_std_unfilt

def maxScoreExperiments(exp_mean_unfilt, exp_mean_filt):
    '''
    maxScoreExperiments - Util
    Description - This function calculates the max experiment configuration in which the highest
                    classification score was recived for filtered and unfiltered data and returns
                    a flag for which ever one got higher score
    :param exp_mean_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_mean_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :return:
            max_score_filt,
            max_exp_filt,
            max_dim_filt,
            max_eps_filt,
            max_neigh_filt, \
            max_score_unfilt,
            max_exp_unfilt,
            max_dim_unfilt,
            max_eps_unfilt,
            max_neigh_unfilt,
            max_filtOrUn - 0 for unfiltered score is gigher, 1 otherwise
    '''
    # Setting variables for max score
    max_score_filt = 0.0
    max_exp_filt = 0
    max_dim_filt = 0
    max_eps_filt = 0
    max_neigh_filt = 0
    max_score_unfilt = 0.0
    max_exp_unfilt = 0
    max_dim_unfilt = 0
    max_eps_unfilt = 0
    max_neigh_unfilt = 0
    max_filtOrUn = 0  # 0 for unfilt, 1 for filt
    # Calculate max scores
    for i in range(N_experiments):
        for j in range(N_dim_parameter):
            for k in range(N_epsilon_parameter):
                for l in range(N_neibours_parameter):
                    if (exp_mean_filt[i][j][k][l] > max_score_filt):
                        max_score_filt = exp_mean_filt[i][j][k][l]
                        max_exp_filt = i
                        max_dim_filt = j
                        max_eps_filt = k
                        max_neigh_filt = l
                    if (exp_mean_unfilt[i][j][k][l] > max_score_unfilt):
                        max_score_unfilt = exp_mean_unfilt[i][j][k][l]
                        max_exp_unfilt = i
                        max_dim_unfilt = j
                        max_eps_unfilt = k
                        max_neigh_unfilt = l

    if (max_score_filt>max_score_unfilt):
        max_filtOrUn = 1
    return  max_score_filt, max_exp_filt, max_dim_filt, max_eps_filt, max_neigh_filt, \
            max_score_unfilt, max_exp_unfilt, max_dim_unfilt, max_eps_unfilt, max_neigh_unfilt, max_filtOrUn

def maxScoreExperimentMeans(mean_exp_mean_unfilt, mean_exp_mean_filt):
    '''
    maxScoreExperimentMeans - Util
    Description - This function calculates the max experiment configuration in which the highest
                    classification score was recived for mean filtered and unfiltered data and returns
                    a flag for which ever one got higher score
    :param mean_exp_mean_unfilt[dim_ind][epsilon_ind][KNNeibours_ind]:
    :param mean_exp_mean_filt[dim_ind][epsilon_ind][KNNeibours_ind]:
    :return:
            max_score_filt,
            max_dim_filt,
            max_eps_filt,
            max_neigh_filt, \
            max_score_unfilt,
            max_dim_unfilt,
            max_eps_unfilt,
            max_neigh_unfilt,
            max_filtOrUn - 0 for unfiltered score is gigher, 1 otherwise
    '''
    # Setting variables for max score
    max_score_filt = 0.0
    max_dim_filt = 0
    max_eps_filt = 0
    max_neigh_filt = 0
    max_score_unfilt = 0.0
    max_dim_unfilt = 0
    max_eps_unfilt = 0
    max_neigh_unfilt = 0
    max_filtOrUn = 0  # 0 for unfilt, 1 for filt
    # Calculate max scores
    for j in range(N_dim_parameter):
        for k in range(N_epsilon_parameter):
            for l in range(N_neibours_parameter):
                if (mean_exp_mean_filt[j][k][l] > max_score_filt):
                    max_score_filt = mean_exp_mean_filt[j][k][l]
                    max_dim_filt = j
                    max_eps_filt = k
                    max_neigh_filt = l
                if (mean_exp_mean_unfilt[j][k][l] > max_score_unfilt):
                    max_score_unfilt = mean_exp_mean_unfilt[j][k][l]
                    max_dim_unfilt = j
                    max_eps_unfilt = k
                    max_neigh_unfilt = l

    if (max_score_filt>max_score_unfilt):
        max_filtOrUn = 1
    return  max_score_filt, max_dim_filt, max_eps_filt, max_neigh_filt, \
            max_score_unfilt, max_dim_unfilt, max_eps_unfilt, max_neigh_unfilt, max_filtOrUn


# ===== Prep ===== #

def get_eeg_o_c_b(eeg_in, half_avg_win):
    '''
    get_eeg_o_c_b - Prep
    Description: Slicing the eeg signal to Open, Closed, Blink, and averaging, for each,
    the surrounding timeframes
    ** CONSTANTS ARE ACCORDING TO SPECIFIC EXPERIMENT SETTINGS **
    :param eeg_in: eeg input signal
    :param half_avg_win: distance (in frames) to average signal on
    :return: list of [eeg_in[open], eeg_in[closed], eeg_in[blinking] ]
    '''
    if(FREQUENCY_ANALYSIS_METHOD=='fft'):
        frames_to_second = int(1 / (TIME_FRAME * (1 - OVERLAP)))
    elif (FREQUENCY_ANALYSIS_METHOD=='wclch'):
        frames_to_second = int(1 / (WELCH_TIME_FRAME * (1 - OVERLAP)))
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
    get_aviv_exp_timeframes - Prep
    Description: extracting labels and legend for Aviv experiment According to experiment specific parameters
    :param eeg_in: frames list -used only for length calculation
    :return: labels, legend
    '''

    session_len = len(eeg_in)
    mode_time_frm = mode_time_frame()

    samples_diff = int(SAMPLES_PER_FRAME*(1-OVERLAP))
    samples_per_mode = int(float(mode_time_frm * FS) / (samples_diff))
    labels = np.zeros(session_len)
    if EXPERIMENT_TYPE == 1:
        legend = {0: 'open',
                  1: 'blink',
                  2: 'closed'}
        start_indices = range(samples_per_mode, session_len, 2 * samples_per_mode)
        for j, start_ind in enumerate(start_indices):
            if start_ind + samples_per_mode >= session_len:
                if j % 2 == 0:
                    labels[start_ind:] = 1
                else:
                    labels[start_ind:] = 2
                continue
            if j % 2 == 0:
                # asserting blink label
                labels[start_ind:start_ind+samples_per_mode] = 1
            else:
                # asserting closed
                labels[start_ind:start_ind + samples_per_mode] = 2
    elif EXPERIMENT_TYPE==2 or EXPERIMENT_TYPE==3:
        legend = {0: 'open',
                  2: 'closed'}
        start_indices = range(0, session_len, samples_per_mode)
        for j, start_ind in enumerate(start_indices):
            if start_ind + samples_per_mode >= session_len:
                if j % 2 == 0:
                    labels[start_ind:] = 0
                else:
                    labels[start_ind:] = 2
                continue
            if j % 2 == 0:
                # asserting open
                labels[start_ind:start_ind+samples_per_mode] = 0
            else:
                # asserting closed
                labels[start_ind:start_ind + samples_per_mode] = 2
    return labels, legend


# ===== Preprocessing ===== #

def butter_bandpass(lowcut, highcut, fs, order=2):
    '''
    Function: butter_bandpass - preprocessing
    Description: returns the buttersworth coefficient for the band pass filter
    :param lowcut: low bandppass frequency [Hz]
    :param highcut: high band pass frequency [Hz]
    :param fs: Sample rate
    :param order: order of the created Buttersworth filter
    :return: filter coefficients a, b (consistant with the fir system representation)
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # print(str(high), '  ', str(low))
    b, a = signal.butter(order, [low, high], btype='band')
    #b, a = signal.butter(order, low)
    return b, a

def fir_bandpass(lowcut=LOW_CUT, highcut=HIGH_CUT, fs=FS, order=ORDER):
    '''
    Function: fir_bandpass - Preprocessing
    Description: returns the FIR filter coefficients for the band pass filter
    :param lowcut: low bandpass frequency
    :param highcut: high bandpass frequency
    :param fs: sample rate
    :param order: order of the constructed FIR filter
    :return: filter h
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    h = signal.firwin(order, [low, high], pass_zero=False)
    return h

def butter_bandpass_filter(data, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=FS, order=ORDER, plotPlease=False):
    '''
    ##Function: butter_bandpass_filter - Preprocessing
    ##Description: filters the date by the produced buttersworth band pass filter
    ##Parameters: data, lowcut, highcut - cutoff frequencies, fs- sample rate, order of filter
    ##Return: filtered data
    :param data:
    :param lowcut: cutoff frequencies
    :param highcut: cutoff frequencies
    :param fs: sample rate
    :param order: order of filter
    :param plotPlease: plot the filter h(t)
    :return: filtered data
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if plotPlease:
        x_tick = np.arange(0., FS / 2, 0.25)
        w, h = signal.freqz(b, a)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label='buterworth order = %d' % order)
        plt.title('Filter - frequency domain')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('abs(Filter amplitude)')
        #plt.show()
        if (SAVE_FIGS == 1):
            plt.savefig(os.path.join(PATH_RESULTS_DIR, FILTER_ANALYSIS_OUTPUT_FILE))
        plt.close()
    y = signal.lfilter(b, a, data, axis=0)
    return y

def load_and_filter_data(path, filter=True,plotPlease=False,meth='fft'):
    '''
    load_and_filter_data - Preprocessing
    Description: loads the file in the specific path,
                 dividing it to windows according to global parameters,
                 and performing FFT + Filtering of the signal
    :param path: path of the file to load data from
    :param filter: flag - 1 - filter, 0 - don't filter
    :param plotPlease: flag plot filter or not
    :param meth: frequency analysis method "fft"/"welch"
    :return: FFT unfiltered, FFT filtered / eeg_segments
    '''
    eeg_data = np.loadtxt(path, comments='%', skiprows=7, usecols=ACTIVE_CHANNELS, delimiter=',')
    start_diff = int(SAMPLES_PER_FRAME*(1- OVERLAP))
    nWindows = int(np.floor(eeg_data.shape[0] / start_diff)-1)

    if (meth=='fft'):
        eeg_fft_filtered = []
        eeg_fft_unfilt = []
        for i in range(nWindows):
            if filter:
                eeg_fft_unfilt.append(np.fft.fftshift(fftpack.fft(sub_mean(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME, :])),axes=1))
                eeg_data_centered = sub_mean(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME,:])  # - np.matlib.repmat(np.mean(eeg_data[i*start_diff:i*start_diff+SAMPLES_PER_FRAME,:], axis=0),SAMPLES_PER_FRAME, 1)
                if (plotPlease==True):
                    if (i==0):
                        eeg_fft_filtered.append(np.fft.fftshift(fftpack.fft(butter_bandpass_filter(eeg_data_centered, plotPlease=True)), axes=1 ))
                    else:
                        eeg_fft_filtered.append(np.fft.fftshift(fftpack.fft(butter_bandpass_filter(eeg_data_centered, plotPlease=False)), axes=1))
                else:
                    eeg_fft_filtered.append(np.fft.fftshift(fftpack.fft(butter_bandpass_filter(eeg_data_centered, plotPlease=False)), axes=1))
            else:
                eeg_fft_unfilt.append(sub_mean(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME, :]))
        if (filter==True):
            return eeg_fft_unfilt, eeg_fft_filtered
        else:
            return eeg_fft_unfilt
    elif (meth=='welch'):
        eeg_welch_filtered = []
        eeg_welch_unfilt = []
        for i in range(nWindows):
            if filter:
                eeg_data_centered = sub_mean(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME,:])
                eeg_welch_unfilt_cur =[]
                for k in range(len(ACTIVE_CHANNELS)):
                    channel = eeg_data_centered.transpose()[k]
                    f_ax, eeg_welch_unfilt_cur_channel = signal.welch(channel.transpose(), float(FS), 'hann',nperseg=int(len(channel.transpose()) / WELCH_SEGMENTS))
                    #f_ax, eeg_welch_unfilt_cur=signal.welch(sub_mean(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME, k]),float(FS),'hann',nperseg=int(len(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME, :])/WELCH_SEGMENTS))
                    eeg_welch_unfilt_cur.append(eeg_welch_unfilt_cur_channel)
                eeg_welch_unfilt.append(eeg_welch_unfilt_cur)
                if (plotPlease==True):
                    if (i==0):
                        eeg_welch_filt_cur = []
                        for k in range(len(ACTIVE_CHANNELS)):
                            if (k==0):
                                channel = eeg_data_centered.transpose()[k]
                                f_ax, eeg_welch_filt_cur_channel =signal.welch(butter_bandpass_filter(channel.transpose(), plotPlease=True),float(FS),'hann',nperseg=int(len(channel.transpose())/WELCH_SEGMENTS))
                                eeg_welch_filt_cur.append(eeg_welch_filt_cur_channel)
                            else:
                                channel = eeg_data_centered.transpose()[k]
                                f_ax, eeg_welch_filt_cur_channel = signal.welch(butter_bandpass_filter(channel.transpose(), plotPlease=False), float(FS), 'hann',nperseg=int(len(channel.transpose()) / WELCH_SEGMENTS))
                                eeg_welch_filt_cur.append(eeg_welch_filt_cur_channel)
                        eeg_welch_filtered.append(eeg_welch_filt_cur)
                    else:
                        eeg_welch_filt_cur=[]
                        for k in range(len(ACTIVE_CHANNELS)):
                            channel = eeg_data_centered.transpose()[k]
                            f_ax, eeg_welch_filt_cur_channel =signal.welch(butter_bandpass_filter(channel.transpose(), plotPlease=False),float(FS),'hann',nperseg=int(len(channel.transpose())/WELCH_SEGMENTS))
                            eeg_welch_filt_cur.append(eeg_welch_filt_cur_channel)
                        eeg_welch_filtered.append(eeg_welch_filt_cur)
                else:
                    eeg_welch_filt_cur = []
                    for k in range(len(ACTIVE_CHANNELS)):
                        channel = eeg_data_centered.transpose()[k]
                        f_ax, eeg_welch_filt_cur_channel =signal.welch(butter_bandpass_filter(channel.transpose(), plotPlease=False),float(FS),'hann',nperseg=int(len(channel.transpose())/WELCH_SEGMENTS))
                        eeg_welch_filt_cur.append(eeg_welch_filt_cur_channel)
                    eeg_welch_filtered.append(eeg_welch_filt_cur)
            else:
                eeg_welch_unfilt.append(sub_mean(eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME, :]))
        if (filter==True):
            return f_ax, eeg_welch_unfilt, eeg_welch_filtered
        else:
            return eeg_welch_unfilt

def preprocess_data(eeg_data, cut_band,f_ax=[]):
    '''
    preprocess_data - Preprocessing
    Description: cuts frequency vectors according to the cut_band so rid of any irrelevant frequency contents for diffusion training
    :param eeg_data: data input (in frequency domain!)
    :param cut_band: 2 values - [freq_low, freq_high]  [Hz]
    :return: eeg_cut: processed data - cutted values (NOT FILTERED) between [ cut_band[0] , cut_band[1] ]
    '''
    if(FREQUENCY_ANALYSIS_METHOD=='fft'):
        num_samples_per_window = FS * TIME_FRAME
        freq_high_per_window = int(FS / 2)
        delta_f = freq_high_per_window / num_samples_per_window
        cut_band_ind_low = int(cut_band[0] / delta_f)
        cut_band_ind_high = int(cut_band[1] / delta_f)
        eeg_cut = [x[cut_band_ind_low:cut_band_ind_high, :] for x in eeg_data]
    elif(FREQUENCY_ANALYSIS_METHOD=='welch'):
        f_ax = np.arange(0, FS / 2, FS / (2 * len(eeg_data[0][0])))
        start_indices = f_ax > cut_band[0]
        end_indices = f_ax < cut_band[1]
        start_ind=0
        while(not(start_indices[start_ind])):
            start_ind=start_ind+1
            end_indices = f_ax < cut_band[1]
        end_ind = len(f_ax)-1
        while (not (end_indices[end_ind])):
            end_ind = end_ind - 1
        eeg_cut=[]
        for i in range(len(eeg_data)):
            eeg_cut_cur =[eeg_data[i][j][start_ind - 1:end_ind + 1] for j in range(len(eeg_data[i]))]
            eeg_cut.append(eeg_cut_cur)
    return eeg_cut


# ===== Analysis ===== #

def knn_clustering(data, labels, neighbors_num=N_NEIGHBORS_LOW):
    '''
    knn_clustering - Analysis
    Description: run KNN clustering on the data and calculate the prediction rate (mean) and std
    :param data: data coordinates - diffusion space coordinates in our case
    :param labels: labels list od data frame cordinates
    :param neighbors_num: number onf neighbours in knn clustering
    :return: prediction score, prediction_std
    '''
    accuracy = []
    neigh = knn(n_neighbors=neighbors_num)
    for d, l in zip(data, labels):
        combined = list(zip(d, l))
        np.random.shuffle(combined)
        d,l = zip(*combined)
        score = cross_val_score(estimator=neigh, X=d, y=l, cv=5)
        accuracy = accuracy + list(score)
    pred_mean = np.mean(accuracy)
    pred_std = np.std(accuracy)
    return pred_mean, pred_std

def svm_cross_val(data, labels):
    '''
    svm_cross_val - Analysis
    Description; run SVM classifier on the data and return the classification scores and std
    :param data: data frames list in frequency domain (welch/fft)
    :param labels: data frames labels list
    :return: prediction SVM score, prediction SVM standrad diviation
    '''
    accuracy = []
    clf = svm.SVC(kernel='rbf')
    combined = list(zip(data, labels))
    np.random.shuffle(combined)
    d, l = zip(*combined)
    score = cross_val_score(clf, X=d, y=l, cv=5)
    accuracy = accuracy + list(score)
    pred_mean = np.mean(accuracy)
    pred_std = np.std(accuracy)
    return pred_mean, pred_std

def pair_wise_knn(data, labels, nn=3):
    '''
    pair_wise_knn - Analysis
    Description: run KNN clustering on 0 vs 2 (open vs closed) only for 3 states analysis of experiment 1 config
    :param data: diffusion space coordinates to be classified according to KNN
    :param labels: data frames labels list
    :return: 2 states prediction score, 2 states standard diviation
    '''
    data_0, labels_0 = extract_state(data, labels, 0)
    # data_1, labels_1 = extract_state(data, labels, 1)
    data_2, labels_2 = extract_state(data, labels, 2)
    data_02 = [d0 + d2 for d0, d2 in zip(data_0, data_2)]
    labels_02 = [l0 + l2 for l0, l2 in zip(labels_0, labels_2)]
    pred_0v2, std_0v2 = knn_clustering(data_02, labels_02, neighbors_num=nn)
    return (pred_0v2, std_0v2)


# ===== Visualizations ===== #

def show_example_freq(eeg_ex_list, config_string,triats_name='',filtOrUnfilt=0):
    '''
    show_example_freq - Visualizations
    Description: Visualizing results of experiment in the frequency domain at 8 time points at the middle of each mode +- EXAMPLES_VISUAL_WINDOW
    :param eeg_ex_list: eeg frames in fft frequency representation
    :param config_string: configuration string for experiment 1 visualizations when SKIP_BLINK is turned off
    :param triats_name: string of triat name
    :param filtOrUnfilt: flag - 0 - unfiltered data, 1 - filtered data
    :return: na
    '''
    if (SKIP_BLINK==1):

        session_len = len(eeg_ex_list)
        mode_time_frm = mode_time_frame()
        samples_diff = int(SAMPLES_PER_FRAME * (1 - OVERLAP))
        samples_per_mode = int(float(mode_time_frm * FS) / (samples_diff))
        index_list_open = range(int(samples_per_mode/2), int(samples_per_mode/2) + 2*samples_per_mode*2,2*samples_per_mode)
        index_list_closed = range(int(samples_per_mode/2)+samples_per_mode, int(samples_per_mode/2) + samples_per_mode + 2*samples_per_mode*2,2*samples_per_mode)
        open_ind=0
        pic_ind =0
        closed_ind=0

        if filtOrUnfilt==0:
            unfilt_filt_str = UNFILTERED_STRING
        else:
            unfilt_filt_str = FILTERED_STRING

        colors = ['red', 'green', 'blue','black','purple']
        for i in range(2):
            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')

            time_point_sec=(index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            #axs=fig.add_subplot(4,4,open_ind+closed_ind+1)
            max_y=0
            min_y=0
            for j in range((eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):]).shape[1]):
                x=np.arange(0 , FS / 2, FS / (2*len(eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME/2):])))
                y=20*np.log10(np.power(np.abs([eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2)+k][j] for k in range((eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):]).shape[0])]),2))
                plt.plot(x,y,c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                cur_max_y=max(y)
                cur_min_y = min(y)
                if j==0:
                    max_y=cur_max_y
                    min_y=cur_min_y
                if cur_max_y>max_y:
                    max_y=cur_max_y
                if cur_min_y<min_y:
                    min_y=cur_min_y
            plt.ylim(0.95*min_y,1.05*max_y)
            plt.title(FFT_STRING+' of eeg for Open state - ' + str(pic_ind)+ ' - ' +unfilt_filt_str + ' Power spectral density - ' + str(time_point_min_int) + ':' + str(time_point_sec_int) + '\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + FILTERED_STRING + '_'+ triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_frequency', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1

            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')
            time_point_sec=(index_list_open[open_ind]+EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            max_y = 0
            min_y = 0
            for j in range((eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):]).shape[1]):
                x=np.arange(0, FS / 2, FS / (2 * len(eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):])))
                y=20 * np.log10(np.power(np.abs([eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2) + k][j] for k in range((eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):]).shape[0])]),2))
                plt.plot(x,y, c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title(FFT_STRING+' of eeg for Open state - '+str(pic_ind)+' - ' +unfilt_filt_str +' Power spectral density - '+str(time_point_min_int)+':'+str(time_point_sec_int)+'\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + FILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_frequency', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1
            open_ind = open_ind + 1

        for i in range(2):
            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')
            time_point_sec=(index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            max_y = 0
            min_y = 0
            for j in range((eeg_ex_list[index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):]).shape[1]):
                x=np.arange(0 , FS / 2, FS / (2*len(eeg_ex_list[index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME/2):])))
                y=20*np.log10(np.power(np.abs([eeg_ex_list[index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2)+k][j] for k in range((eeg_ex_list[index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):]).shape[0])]),2))
                plt.plot(x,y,c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title(FFT_STRING+'of eeg for Closed state - '+str(pic_ind)+' - ' +unfilt_filt_str +' Power spectral density - '+str(time_point_min_int)+':'+str(time_point_sec_int)+'\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + FILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Closed.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(time_point_sec_int) + '_Closed.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_frequency', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1

            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')
            time_point_sec=(index_list_closed[closed_ind]+EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            max_y = 0
            min_y = 0
            for j in range((eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):]).shape[1]):
                x=np.arange(0, FS / 2, FS / (2 * len(eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):])))
                y=20 * np.log10(np.power(np.abs([eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2) + k][j] for k in range((eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][int(SAMPLES_PER_FRAME / 2):]).shape[0])]),2))
                plt.plot(x,y, c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title(FFT_STRING+'of eeg for Closed state - ' + str(pic_ind) +' - ' +unfilt_filt_str + ' Power spectral density - ' + str(time_point_min_int) + ':' + str(time_point_sec_int) + '\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + FILTERED_STRING + '_'+triats_name + '_ind_' + str(pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(time_point_sec_int) + '_Closed.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Closed.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_frequency', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1
            closed_ind = closed_ind + 1

    else:
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
        #plt.show()
        if (SAVE_FIGS == 1):
            plt.savefig(os.path.join(PATH_RESULTS_DIR, SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ))
        plt.close()
    return

def show_example_freq_welch(eeg_ex_list, config_string,triats_name='',filtOrUnfilt=0):
    '''
    show_example_freq_welch - Visualizations
    Description: Visualizing results of experiment in the frequency domain at 8 time points at the middle of each mode +- EXAMPLES_VISUAL_WINDOW
    :param eeg_ex_list: eeg frames in Welch frequency representation
    :param config_string: configuration string for experiment 1 visualizations when SKIP_BLINK is turned off
    :param triats_name: string of triat name
    :param filtOrUnfilt: flag - 0 - unfiltered data, 1 - filtered data
    :return: na
    '''
    if (SKIP_BLINK==1):

        session_len = len(eeg_ex_list)
        mode_time_frm = mode_time_frame()
        samples_diff = int(SAMPLES_PER_FRAME * (1 - OVERLAP))
        samples_per_mode = int(float(mode_time_frm * FS) / (samples_diff))
        index_list_open = range(int(samples_per_mode/2), int(samples_per_mode/2) + 2*samples_per_mode*2,2*samples_per_mode)
        index_list_closed = range(int(samples_per_mode/2)+samples_per_mode, int(samples_per_mode/2) + samples_per_mode + 2*samples_per_mode*2,2*samples_per_mode)
        open_ind=0
        pic_ind =0
        closed_ind=0

        if filtOrUnfilt==0:
            unfilt_filt_str = UNFILTERED_STRING
        else:
            unfilt_filt_str = FILTERED_STRING

        colors = ['red', 'green', 'blue','black','purple']
        for i in range(2):
            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')

            time_point_sec=(index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            #axs=fig.add_subplot(4,4,open_ind+closed_ind+1)
            f_ax = np.arange(0,FS/2,FS/(2*len(eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][0])))
            max_y = 0
            min_y = 0
            for j in range(len(eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW])):
                y=20*np.log10(np.abs([eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][j][k] for k in range(len(eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][j]))]))
                plt.plot(f_ax,y,c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title('Welch frequency analysis of eeg for Open state - ' + str(pic_ind) + ' - ' +unfilt_filt_str + '\nPower spectral density - ' + str(time_point_min_int) + ':' + str(time_point_sec_int) + '\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ +WELCH_STRING+ '_'+FILTERED_STRING + '_'+ triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ+WELCH_STRING + '_'+ UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_frequency', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1

            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')

            time_point_sec = (index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW) * samples_diff / FS
            time_point_min_int = int(time_point_sec / 60)
            time_point_sec_int = int(time_point_sec % 60)
            # axs=fig.add_subplot(4,4,open_ind+closed_ind+1)
            f_ax = np.arange(0, FS / 2, FS / (2 * len(eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][0])))
            max_y = 0
            min_y = 0
            for j in range(len(eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW])):
                y=20 * np.log10(np.abs([eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][j][k] for k in range(len(eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][j]))]))
                plt.plot(f_ax, y, c=colors[j], label='Channel ' + str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title('Welch frequency analysis of eeg for Open state - ' + str(pic_ind) + ' - ' +unfilt_filt_str + '\nPower spectral density - ' + str(time_point_min_int) + ':' + str(time_point_sec_int) + '\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + WELCH_STRING + '_' + FILTERED_STRING + '_' + triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + WELCH_STRING + '_' + UNFILTERED_STRING + '_' + triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_frequency', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1
            open_ind = open_ind + 1

        for i in range(2):
            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')

            time_point_sec = (index_list_closed[closed_ind] - EXAMPLES_VIZUAL_WINDOW) * samples_diff / FS
            time_point_min_int = int(time_point_sec / 60)
            time_point_sec_int = int(time_point_sec % 60)
            # axs=fig.add_subplot(4,4,open_ind+closed_ind+1)
            f_ax = np.arange(0, FS / 2, FS / (2 * len(eeg_ex_list[index_list_closed[closed_ind] - EXAMPLES_VIZUAL_WINDOW][0])))
            max_y = 0
            min_y = 0
            for j in range(len(eeg_ex_list[index_list_closed[closed_ind] - EXAMPLES_VIZUAL_WINDOW])):
                y=20 * np.log10(np.abs([eeg_ex_list[index_list_closed[closed_ind] - EXAMPLES_VIZUAL_WINDOW][j][k] for k in range(len(eeg_ex_list[index_list_closed[closed_ind] - EXAMPLES_VIZUAL_WINDOW][j]))]))
                plt.plot(f_ax, y,c=colors[j], label='Channel ' + str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title('Welch frequency analysis of eeg for Closed state - ' + str(pic_ind) + ' - ' +unfilt_filt_str + '\nPower spectral density - ' + str(time_point_min_int) + ':' + str(time_point_sec_int) + '\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + WELCH_STRING + '_' + FILTERED_STRING + '_' + triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Closed.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + WELCH_STRING + '_' + UNFILTERED_STRING + '_' + triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Closed.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_frequency', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1

            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')

            time_point_sec = (index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW) * samples_diff / FS
            time_point_min_int = int(time_point_sec / 60)
            time_point_sec_int = int(time_point_sec % 60)
            # axs=fig.add_subplot(4,4,open_ind+closed_ind+1)
            f_ax = np.arange(0, FS / 2, FS / (2 * len(eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][0])))
            max_y = 0
            min_y = 0
            for j in range(len(eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW])):
                y=20 * np.log10(np.abs([eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][j][k] for k in range(len(eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][j]))]))
                plt.plot(f_ax, y, c=colors[j], label='Channel ' + str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title('Welch frequency analysis of eeg for Closed state - ' + str(pic_ind) + ' - ' +unfilt_filt_str + '\nPower spectral density - ' +str(time_point_min_int) + ':' + str(time_point_sec_int) + '\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + WELCH_STRING + '_' + FILTERED_STRING + '_' + triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Closed.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ + WELCH_STRING + '_' + UNFILTERED_STRING + '_' + triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Closed.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_frequency', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1
            closed_ind = closed_ind + 1
    else:
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
        #plt.show()
        if (SAVE_FIGS == 1):
            plt.savefig(os.path.join(PATH_RESULTS_DIR, SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ))
        plt.close()
    return

def show_example_freq_avg(eeg_ex_list, config_string,triats_name='',filtOrUnfilt=0,half_win=0):
    '''
    show_example_freq_avg - Visualizations
    Description: Visualizing results of experiment averaging +- (half_wim-1) (parameter) frames in the frequency domain at 8 time points at the middle of each mode +- EXAMPLES_VISUAL_WINDOW
    :param eeg_ex_list: eeg frames segments in time domain
    :param config_string: configuration string for experiment 1 visualizations when SKIP_BLINK is turned off
    :param triats_name: string of triat name
    :param filtOrUnfilt: flag - 0 - unfiltered data, 1 - filtered data
    :param half_win: averaging window
    :return: na
    '''
    if (SKIP_BLINK==1):

        session_len = len(eeg_ex_list)
        mode_time_frm = mode_time_frame()
        samples_diff = int(SAMPLES_PER_FRAME * (1 - OVERLAP))
        samples_per_mode = int(float(mode_time_frm * FS) / (samples_diff))
        index_list_open = range(int(samples_per_mode/2), int(samples_per_mode/2) + 2*samples_per_mode*2,2*samples_per_mode)
        index_list_closed = range(int(samples_per_mode/2)+samples_per_mode, int(samples_per_mode/2) + samples_per_mode + 2*samples_per_mode*2,2*samples_per_mode)
        open_ind=0
        pic_ind =0
        closed_ind=0

        if filtOrUnfilt==0:
            unfilt_filt_str = UNFILTERED_STRING
        else:
            unfilt_filt_str = FILTERED_STRING

        colors = ['red', 'green', 'blue','black','purple']
        for i in range(2):
            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')

            time_point_sec=(index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            #axs=fig.add_subplot(4,4,open_ind+closed_ind+1
            eeg_avg=avg_timeframe(eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW-half_win:index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW+half_win])
            max_y = 0
            min_y = 0
            for j in range((eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):]).shape[1]):
                x=np.arange(0 , FS / 2, FS / (2*len(eeg_avg[int(np.ceil(SAMPLES_PER_FRAME/2)):])))
                y=20*np.log10(np.power(np.abs([eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2))+k][j] for k in range((eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):]).shape[0])]),2))
                plt.plot(x,y,c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(max(0,0.95*min_y), 1.05 * max_y)
            plt.title('Frequency analysis average of eeg for Open states - ' + str(pic_ind) + ' - ' +unfilt_filt_str + '\nPower spectral density - ' + str(time_point_min_int) + ':' + str(time_point_sec_int) + ' +- '+str(half_win-1)+ ' frames\n')
            axes = plt.gca()
            axes.set_xlim([0,50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ_AVG + FILTERED_STRING + '_'+ triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ_AVG + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_averages', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1

            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')
            time_point_sec=(index_list_open[open_ind]+EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            eeg_avg=avg_timeframe(eeg_ex_list[index_list_open[open_ind]+EXAMPLES_VIZUAL_WINDOW-half_win:index_list_open[open_ind]+EXAMPLES_VIZUAL_WINDOW+half_win])
            max_y = 0
            min_y = 0
            for j in range((eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):]).shape[1]):
                x=np.arange(0, FS / 2, FS / (2 * len(eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):])))
                y=20 * np.log10(np.power(np.abs([eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2))+ k][j] for k in range((eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):]).shape[0])]),2))
                plt.plot(x,y, c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(max(0, 0.95 * min_y), 1.05 * max_y)
            plt.title('Frequency analysis of average eeg for Open states - '+str(pic_ind)+' - ' +unfilt_filt_str + '\nPower spectral density - '+str(time_point_min_int)+':'+str(time_point_sec_int)+ ' +- '+str(half_win-1)+ ' frames\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ_AVG + FILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ_AVG + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                    plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_averages', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1
            open_ind = open_ind + 1

        for i in range(2):
            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')
            time_point_sec=(index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            eeg_avg=avg_timeframe(eeg_ex_list[index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW-half_win:index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW+half_win])
            max_y = 0
            min_y = 0
            for j in range((eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):]).shape[1]):
                x=np.arange(0 , FS / 2, FS / (2*len(eeg_avg[int(np.ceil(SAMPLES_PER_FRAME/2)):])))
                y=20*np.log10(np.power(np.abs([eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2))+k][j] for k in range((eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):]).shape[0])]),2))
                plt.plot(x,y,c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(max(0, 0.95 * min_y), 1.05 * max_y)
            plt.title('Frequency analysis of average eeg for Closed states - '+str(pic_ind)+' - ' +unfilt_filt_str + '\nPower spectral density - '+str(time_point_min_int)+':'+str(time_point_sec_int)+ ' +- '+str(half_win-1)+ ' frames\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ_AVG + FILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Closed.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ_AVG + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(time_point_sec_int) + '_Closed.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_averages', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1

            fig = plt.figure()
            plt.xlabel('F [Hz]')
            plt.ylabel('|Amp[uV]|^2 [dB]')
            time_point_sec=(index_list_closed[closed_ind]+EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            eeg_avg=avg_timeframe(eeg_ex_list[index_list_closed[closed_ind]+EXAMPLES_VIZUAL_WINDOW-half_win:index_list_closed[closed_ind]+EXAMPLES_VIZUAL_WINDOW+half_win])
            max_y = 0
            min_y = 0
            for j in range((eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):]).shape[1]):
                x=np.arange(0, FS / 2, FS / (2 * len(eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):])))
                y=20 * np.log10(np.power(np.abs([eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2) )+ k][j] for k in range((eeg_avg[int(np.ceil(SAMPLES_PER_FRAME / 2)):]).shape[0])]),2))
                plt.plot(x,y, c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                cur_max_y = max(y)
                cur_min_y = min(y)
                if j == 0:
                    max_y = cur_max_y
                    min_y = cur_min_y
                if cur_max_y > max_y:
                    max_y = cur_max_y
                if cur_min_y < min_y:
                    min_y = cur_min_y
            plt.ylim(max(0, 0.95 * min_y), 1.05 * max_y)
            plt.title('Frequency analysis of average eeg for Closed states - ' + str(pic_ind) + ' - ' +unfilt_filt_str + '\nPower spectral density - ' + str(time_point_min_int) + ':' + str(time_point_sec_int) + ' +- '+str(half_win-1)+ ' frames\n')
            axes = plt.gca()
            axes.set_xlim([0, 50])
            plt.legend()
            if (SAVE_FIGS == 1):
                if (filtOrUnfilt == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ_AVG + FILTERED_STRING + '_'+triats_name + '_ind_' + str(pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(time_point_sec_int) + '_Closed.png'
                else:
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ_AVG + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Closed.png'
                    plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_averages', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1
            closed_ind = closed_ind + 1

    else:
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
        #plt.show()
        if (SAVE_FIGS == 1):
            plt.savefig(os.path.join(PATH_RESULTS_DIR, SIGNAL_EXAMPLE_OUTPUT_FILE_FREQ))
        plt.close()
    return

def show_example_time(eeg_ex_list,triats_name='',filtOrUnfilt=0):
    '''
    show_example_time - Visualizations
    Description: Visualizing results of experiment  in the time domain at 8 time points at the middle of each mode +- EXAMPLES_VISUAL_WINDOW
    :param eeg_ex_list: eeg frames segments in time domain
    :param config_string: configuration string for experiment 1 visualizations when SKIP_BLINK is turned off
    :param triats_name: string of triat name
    :param filtOrUnfilt: flag - 0 - unfiltered data, 1 - filtered data
    :return: na
    '''
    if (SKIP_BLINK==1):
        session_len = len(eeg_ex_list)
        mode_time_frm = mode_time_frame()
        samples_diff = int(SAMPLES_PER_FRAME * (1 - OVERLAP))
        samples_per_mode = int(float(mode_time_frm * FS) / (samples_diff))
        index_list_open = range(int(samples_per_mode/2), int(samples_per_mode/2) + 2*samples_per_mode*2,2*samples_per_mode)
        index_list_closed = range(int(samples_per_mode/2)+samples_per_mode, int(samples_per_mode/2) + samples_per_mode + 2*samples_per_mode*2,2*samples_per_mode)
        open_ind=0
        pic_ind =0
        closed_ind=0

        if filtOrUnfilt==0:
            unfilt_filt_str = UNFILTERED_STRING
        else:
            unfilt_filt_str = FILTERED_STRING

        colors = ['red', 'green', 'blue','black','purple']
        for i in range(2):
            fig = plt.figure()
            plt.xlabel('Time [s]')
            plt.ylabel('Amp[uV]')

            time_point_sec=(index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            max_y = 0
            min_y = 0
            for j in range((eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][:]).shape[1]):
                if (FREQUENCY_ANALYSIS_METHOD=='fft'):
                    x=np.arange(0 ,TIME_FRAME/TIME_FRAME_PARTIAL, TIME_FRAME / (TIME_FRAME_PARTIAL*len(eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][:])/TIME_FRAME_PARTIAL))
                    y=np.asarray([eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][k][j] for k in range(int((eeg_ex_list[index_list_open[open_ind]-EXAMPLES_VIZUAL_WINDOW][:]).shape[0]/TIME_FRAME_PARTIAL))])
                    plt.plot(x,y,c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                    cur_max_y = max(y)
                    cur_min_y = min(y)
                    if j == 0:
                        max_y = cur_max_y
                        min_y = cur_min_y
                    if cur_max_y > max_y:
                        max_y = cur_max_y
                    if cur_min_y < min_y:
                        min_y = cur_min_y
                elif (FREQUENCY_ANALYSIS_METHOD=='welch'):
                    x =np.arange(0, WELCH_TIME_FRAME/TIME_FRAME_PARTIAL, WELCH_TIME_FRAME / (TIME_FRAME_PARTIAL*len(eeg_ex_list[index_list_open[open_ind] - EXAMPLES_VIZUAL_WINDOW][:])/TIME_FRAME_PARTIAL))
                    y =np.asarray([eeg_ex_list[index_list_open[open_ind] - EXAMPLES_VIZUAL_WINDOW][k][j] for k in range(int((eeg_ex_list[index_list_open[open_ind] - EXAMPLES_VIZUAL_WINDOW][:]).shape[0]/TIME_FRAME_PARTIAL))])
                    plt.plot(x, y,c=colors[j], label='Channel ' + str(ACTIVE_CHANNELS[j]))
                    cur_max_y = max(y)
                    cur_min_y = min(y)
                    if j == 0:
                        max_y = cur_max_y
                        min_y = cur_min_y
                    if cur_max_y > max_y:
                        max_y = cur_max_y
                    if cur_min_y < min_y:
                        min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title('EEG time frame for Open state - ' + str(pic_ind) + ' - ' +unfilt_filt_str +' - ' + str(time_point_min_int) + ':' + str(time_point_sec_int) + '\n')
            plt.legend()
            if (SAVE_FIGS == 1):
                File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_TIME + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                    pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                    time_point_sec_int) + '_Open.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_time', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1

            fig = plt.figure()
            plt.xlabel('Time [s]')
            plt.ylabel('Amp[uV]')
            time_point_sec=(index_list_open[open_ind]+EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            max_y = 0
            min_y = 0
            for j in range((eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][:]).shape[1]):
                if (FREQUENCY_ANALYSIS_METHOD == 'fft'):
                    x =np.arange(0, TIME_FRAME/TIME_FRAME_PARTIAL,TIME_FRAME /(TIME_FRAME_PARTIAL*len(eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][:])/TIME_FRAME_PARTIAL))
                    y =np.asarray([eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][k][j] for k in range(int((eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][:]).shape[0]/TIME_FRAME_PARTIAL))])
                    plt.plot(x,y, c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                    cur_max_y = max(y)
                    cur_min_y = min(y)
                    if j == 0:
                        max_y = cur_max_y
                        min_y = cur_min_y
                    if cur_max_y > max_y:
                        max_y = cur_max_y
                    if cur_min_y < min_y:
                        min_y = cur_min_y
                elif (FREQUENCY_ANALYSIS_METHOD == 'welch'):
                    x = np.arange(0, WELCH_TIME_FRAME/TIME_FRAME_PARTIAL, WELCH_TIME_FRAME / (TIME_FRAME_PARTIAL*len(eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][:])/TIME_FRAME_PARTIAL))
                    y =np.asarray([eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][k][j] for k in range(int((eeg_ex_list[index_list_open[open_ind] + EXAMPLES_VIZUAL_WINDOW][:]).shape[0]/TIME_FRAME_PARTIAL))])
                    plt.plot(x, y,c=colors[j], label='Channel ' + str(ACTIVE_CHANNELS[j]))
                    cur_max_y = max(y)
                    cur_min_y = min(y)
                    if j == 0:
                        max_y = cur_max_y
                        min_y = cur_min_y
                    if cur_max_y > max_y:
                        max_y = cur_max_y
                    if cur_min_y < min_y:
                        min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title('EEG time frame for Open state - '+str(pic_ind)+' - ' +unfilt_filt_str +' - '+str(time_point_min_int)+':'+str(time_point_sec_int)+'\n')
            plt.legend()
            if (SAVE_FIGS == 1):
                    File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_TIME + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                        pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                        time_point_sec_int) + '_Open.png'
                    plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_time', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1
            open_ind = open_ind + 1

        for i in range(2):
            fig = plt.figure()
            plt.xlabel('Time [s]')
            plt.ylabel('Amp[uV]')
            time_point_sec=(index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            max_y = 0
            min_y = 0
            for j in range((eeg_ex_list[index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW][:]).shape[1]):
                if (FREQUENCY_ANALYSIS_METHOD == 'fft'):
                    x = np.arange(0 , TIME_FRAME/TIME_FRAME_PARTIAL,TIME_FRAME/ (len(eeg_ex_list[index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW][:])))
                    y =np.asarray([eeg_ex_list[index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW][k][j] for k in range(int((eeg_ex_list[index_list_closed[closed_ind]-EXAMPLES_VIZUAL_WINDOW][:]).shape[0]/TIME_FRAME_PARTIAL))])
                    plt.plot(x,y,c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                    cur_max_y = max(y)
                    cur_min_y = min(y)
                    if j == 0:
                        max_y = cur_max_y
                        min_y = cur_min_y
                    if cur_max_y > max_y:
                        max_y = cur_max_y
                    if cur_min_y < min_y:
                        min_y = cur_min_y
                elif (FREQUENCY_ANALYSIS_METHOD == 'welch'):
                    x =np.arange(0, WELCH_TIME_FRAME/TIME_FRAME_PARTIAL,WELCH_TIME_FRAME / (len(eeg_ex_list[index_list_closed[closed_ind] - EXAMPLES_VIZUAL_WINDOW][:])))
                    y =np.asarray([eeg_ex_list[index_list_closed[closed_ind] - EXAMPLES_VIZUAL_WINDOW][k][j] for k in range(int((eeg_ex_list[index_list_closed[closed_ind] - EXAMPLES_VIZUAL_WINDOW][:]).shape[0]/TIME_FRAME_PARTIAL))])
                    plt.plot(x,y, c=colors[j],label='Channel ' + str(ACTIVE_CHANNELS[j]))
                    cur_max_y = max(y)
                    cur_min_y = min(y)
                    if j == 0:
                        max_y = cur_max_y
                        min_y = cur_min_y
                    if cur_max_y > max_y:
                        max_y = cur_max_y
                    if cur_min_y < min_y:
                        min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title('EEG time frame for Closed state - '+str(pic_ind)+' - ' +unfilt_filt_str +' - '+str(time_point_min_int)+':'+str(time_point_sec_int)+'\n')
            plt.legend()
            if (SAVE_FIGS == 1):
                File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_TIME + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(time_point_sec_int) + '_Closed.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_time', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1

            fig = plt.figure()
            plt.xlabel('Time [s]')
            plt.ylabel('Amp[uV]')
            time_point_sec=(index_list_closed[closed_ind]+EXAMPLES_VIZUAL_WINDOW)*samples_diff/FS
            time_point_min_int=int(time_point_sec/60)
            time_point_sec_int=int(time_point_sec % 60)
            max_y = 0
            min_y = 0
            for j in range((eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][:]).shape[1]):
                if (FREQUENCY_ANALYSIS_METHOD == 'fft'):
                    x = np.arange(0, TIME_FRAME/TIME_FRAME_PARTIAL, TIME_FRAME / ( len(eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][:])))
                    y =np.asarray([eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][k][j] for k in range(int((eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][:]).shape[0]/TIME_FRAME_PARTIAL))])
                    plt.plot(x,y, c=colors[j],label='Channel '+str(ACTIVE_CHANNELS[j]))
                    cur_max_y = max(y)
                    cur_min_y = min(y)
                    if j == 0:
                        max_y = cur_max_y
                        min_y = cur_min_y
                    if cur_max_y > max_y:
                        max_y = cur_max_y
                    if cur_min_y < min_y:
                        min_y = cur_min_y
                elif (FREQUENCY_ANALYSIS_METHOD == 'welch'):
                    x = np.arange(0, WELCH_TIME_FRAME/TIME_FRAME_PARTIAL,WELCH_TIME_FRAME / (len(eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][:])))
                    y = np.asarray([eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][k][j] for k in range(int((eeg_ex_list[index_list_closed[closed_ind] + EXAMPLES_VIZUAL_WINDOW][:]).shape[0]/TIME_FRAME_PARTIAL))])
                    plt.plot(x,y, c=colors[j],label='Channel ' + str(ACTIVE_CHANNELS[j]))
                    cur_max_y = max(y)
                    cur_min_y = min(y)
                    if j == 0:
                        max_y = cur_max_y
                        min_y = cur_min_y
                    if cur_max_y > max_y:
                        max_y = cur_max_y
                    if cur_min_y < min_y:
                        min_y = cur_min_y
            plt.ylim(0.95 * min_y, 1.05 * max_y)
            plt.title('EEG time frame for Closed state - ' + str(pic_ind) + ' - ' +unfilt_filt_str +' - ' + str(time_point_min_int) + ':' + str(time_point_sec_int) + '\n')
            plt.legend()
            if (SAVE_FIGS == 1):
                File_string_name = SIGNAL_EXAMPLE_OUTPUT_FILE_TIME + UNFILTERED_STRING + '_'+triats_name + '_ind_' + str(
                    pic_ind) + '_time_' + str(time_point_min_int) + '_' + str(
                    time_point_sec_int) + '_Closed.png'
                plt.savefig(os.path.join(CUR_TRIAT_PATH,'Examples_time', File_string_name))
            plt.close()
            pic_ind = pic_ind + 1
            closed_ind = closed_ind + 1

    else:
        fig, ax = plt.subplots(nrows=3, ncols=3)
        plt.suptitle('Frequency analysis of eeg for Open, Closed, Blink states')
       # fig.text(0.5, 0.9, config_string, ha='center')
        fig.text(0.5, 0.04, 'F [Hz]', ha='center')
        fig.text(0.07, 0.5, '|Amp| [dB]', va='center', rotation='vertical')
        fig.text(0.04, 0.23, 'Blink', va='center', rotation='vertical')
        fig.text(0.04, 0.5, 'Closed', va='center', rotation='vertical')
        fig.text(0.04, 0.77, 'Open', va='center', rotation='vertical')
        for i in range(3):
            for j in range(3):
                ax[i, j].plot(20* np.log10(np.abs(eeg_ex_list[i][j][:int(SAMPLES_PER_FRAME/4), :])))
        #plt.show()
        if (SAVE_FIGS == 1):
            plt.savefig(os.path.join(PATH_RESULTS_DIR, SIGNAL_EXAMPLE_OUTPUT_FILE_TIME))
        plt.close()
    return

def get_config_diffusion(exp_ind=0,eps_ind=0,triats_name=''):
    '''
    get_config_diffusion - Visualizations
    Description:
    :param exp_ind:
    :param eps_ind:
    :param triats_name:
    :return:
    '''
    config_string = ' - Config: Eps=' + str(epsilon_array[eps_ind])
    file_config_string = '_Config_Eps_' + str(epsilon_array[eps_ind])
    exp_string = 'Experiment ' + str(exp_ind)+' - '+triats_name
    file_exp_string = 'Experiment_' + str(exp_ind)+'_'+triats_name
    return [exp_string + config_string,file_exp_string + file_config_string]

def show_diffusion(coordinates, labels_list, legend,config_string='', filtOrUnfilt=0):
    '''
    show_diffusion - Visualizations
    Description: Visualizing results of Diffusion maps embedded data for single analysis
    :param coordinates: Diffusion space coordinates list
    :param labels_list: list of data labels
    :param legend: data legend  - states - state_number 
    :param config_string: configuration string for adding a config string to the diffusion title and file name of plot
    :param filtOrUnfilt: flag - 0 - unfiltered data, 1 - filtered data
    :return: na
    '''colors = ['red', 'green', 'blue']
    i = 0
    fig = plt.figure()
    a = np.asarray(coordinates)
    labels = labels_list
    x = a[:, 0]
    if a.shape[1] > 1:
        y = a[:, 1]
    else:
        y = np.zeros(x.shape)
    if a.shape[1] >= 3:
        ax = fig.gca(projection='3d')
        z = a[:, 2]
        for label in legend:
            cur_label = legend[label]
            Axes3D.scatter(ax, np.asarray(x[labels[0]==label]), np.asarray(y[labels[0]==label]),
                           np.asarray(z[labels[0]==label]), c=colors[label], label=cur_label)
    else:
        ax = fig.gca()
        for label in legend:
            cur_label = legend[label]
            ax.scatter(np.asarray(x[labels[0]==label]), np.asarray(y[labels[0]==label]),
                       c=colors[label], label=cur_label)
    i += 1
    if (filtOrUnfilt == 1):
        plot_title = 'Diffusion visualization ' + config_string + FILTERED_STRING
    else:
        plot_title = 'Diffusion visualization ' + config_string + UNFILTERED_STRING
    plt.title(plot_title)
    plt.legend()
    #plt.show()
    if (SAVE_FIGS == 1):
       if (filtOrUnfilt==1):
           File_string_name=DIFFUSION_EXAMPLE_OUTPUT_FILE + config_string + FILTERED_STRING + '.png'
       else:
           File_string_name = DIFFUSION_EXAMPLE_OUTPUT_FILE + config_string + UNFILTERED_STRING + '.png'
       plt.savefig(os.path.join(CUR_TRIAT_PATH,'Diffusion_results' ,File_string_name))
       plt.close()

def show_embedded(coordinates, labels, legend):
    '''
    show_embedded - Visualiztions
    Description: plots a list of data frame embedded coordinates in 2D
    :param coordinates: data frames embedded coordinates
    :param labels: data frames labels list
    :param legend: data frames legend
    :return : na
    '''
    fig = plt.figure()
    colors = ['red', 'green', 'blue']
    a = np.asarray(coordinates)
    x = a[:, 0]
    y = a[:, 1]
    for label in legend:
        cur_label = legend[label]
        plt.scatter(x[labels==label], y[labels==label], c=colors[label], label=cur_label)
    plt.legend()
    plt.show()

def show_3D(X_embedded, labels, fig, meth, domain):
    '''
    show_embedded - Visualiztions
    Description: plots a list of data frame embedded coordinates in 2D
    :param X_embedded: data frames embedded coordinates
    :param labels: data frames labels list
    :param fig: figure object
    :param meth: method in which analysis was made with (PCA,LLE,Tsne,Isomap
    :param domain: frequency/time domains - usually frequency
    :return: na
    '''
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Method = %s, Domain = %s' % (meth, domain))
    for i in np.sort(np.unique(labels)):
        indices= labels == i
        ax.scatter(np.asarray(X_embedded[indices,0]),np.asarray(X_embedded[indices,1]),np.asarray(X_embedded[indices,2]),label=legend[i])
    plt.show()

def show_embedded_all(coordinates_tsne,coordinates_lle,coordinates_pca, coordinates_isomap, labels, legend,exp_ind,triats_name='', filtOrUn=0):
    '''
    show_embedded_all - Vizualizations
    Description: Plot all 4 visualizations method embedded coordiantes of the data frames frequency vectors to a figure
    :param coordinates_tsne:
    :param coordinates_lle:
    :param coordinates_pca:
    :param coordinates_isomap:
    :param labels:
    :param legend:
    :param exp_ind: Experiment number
    :param triats_name: exxperiment triat name
    :param filtOrUn: flag - 1 - filtered data, 0 - unfiltered data
    :return: Na
    '''
    colors = ['red', 'green', 'blue']
    fig = plt.figure(figsize=(9,9))
    ax_221 = fig.add_subplot(221)
    ax_221.set_title('Tsne')
    a = np.asarray(coordinates_tsne)
    x = a[:, 0]
    y = a[:, 1]
    if SKIP_BLINK:
        legend= {0: 'open', 2: 'closed'}
    for label in legend:
        cur_label = legend[label]
        plt.scatter(x[np.asarray(labels)==label], y[np.asarray(labels)==label], c=colors[label], label=cur_label)
    ax_222 = fig.add_subplot(222)
    ax_222.set_title('LLE')
    a = np.asarray(coordinates_lle)
    x = a[:, 0]
    y = a[:, 1]
    for label in legend:
        cur_label = legend[label]
        plt.scatter(x[np.asarray(labels) == label], y[np.asarray(labels) == label], c=colors[label], label=cur_label)
    ax_223 = fig.add_subplot(223)
    ax_223.set_title('PCA')
    a = np.asarray(coordinates_pca)
    x = a[:, 0]
    y = a[:, 1]
    for label in legend:
        cur_label = legend[label]
        plt.scatter(x[np.asarray(labels) == label], y[np.asarray(labels) == label], c=colors[label], label=cur_label)
    ax_224 = fig.add_subplot(224)
    ax_224.set_title('Isomap')
    a = np.asarray(coordinates_isomap)
    x = a[:, 0]
    y = a[:, 1]
    for label in legend:
        cur_label = legend[label]
        plt.scatter(x[np.asarray(labels) == label], y[np.asarray(labels) == label], c=colors[label], label=cur_label)
    plt.legend(loc=2)
    fig_title='2-D Visualization of eeg data'
    exp_string=' Experiment '+str(exp_ind) + ' - '+ triats_name
    file_name_string='Experiment_'+str(exp_ind)+ '_'+triats_name +'_Visualization'
    if(filtOrUn==0):
        fig_title=fig_title+exp_string+' - Unfiltered'
        file_name_string=file_name_string+'_Unfiltered.png'
    else :
        fig_title = fig_title+exp_string + ' - Filtered'
        file_name_string = file_name_string+ '_Filtered.png'
    plt.suptitle(fig_title)
 #   plt.show()
    if(SAVE_FIGS==1):
        plt.savefig(os.path.join(CUR_TRIAT_PATH, 'Visualizations',file_name_string))
    plt.close()

def visualize_eeg(eeg_data, labels, legend, fs, domain='Freq', meth='tsne', dim=2, exp_ind=1,triats_name='', filtOrUn=0):
    '''
    visualize_eeg - Visualizations
    Description: Visualize the input eeg data in time domain
    :param eeg_data: 2d time domain eeg data - [samples, channels]
    :param fs: sample frequency (for axis labels)
    :param labels: labels of eeg dataset according to labels
    :param legend: legend of data and labels tagging
    :param domain: 'Time'/ 'Freq' - the domain we want to visualize in. Note: Data is recived in frequency domain and entering time does not take mean into account 
    :param meth: 'tsne'/ 'lle'/ 'pca'/ 'isomap' - the method used for data embedding in lower dimensional space
    :param dim: 2(default)/ 3 - the dimenssion to show the data in
    :return: None
    '''

    # preform lower dimenstional embbeding and show the results
    if meth=='tsne':
 #       print eeg_data.shape
 #       eeg_data_float64 = np.asarray(eeg_data).astype('float64')
 #       eeg_data_float64 = eeg_data_float64.reshape((eeg_data_float64.shape[0], -1))
        X_embedded=manifold.TSNE(dim).fit_transform(eeg_data)
    elif meth=='lle':
        X_embedded= manifold.LocallyLinearEmbedding(n_neighbors, dim,eigen_solver='auto', method='standard').fit_transform(eeg_data)
    elif meth=='pca':
        X_embedded=decomposition.PCA(dim).fit_transform(eeg_data)
    elif meth=='isomap':
        X_embedded=manifold.Isomap(n_neighbors, dim).fit_transform(eeg_data)
    elif meth=='all':
        X_embedded_tsne=manifold.TSNE(dim).fit_transform(eeg_data)
        X_embedded_lle = manifold.LocallyLinearEmbedding(n_neighbors, dim, eigen_solver='auto',
                                                 method='standard').fit_transform(eeg_data)
        X_embedded_pca = decomposition.PCA(dim).fit_transform(eeg_data)
        X_embedded_isomap = manifold.Isomap(n_neighbors, dim).fit_transform(eeg_data)
        show_embedded_all(X_embedded_tsne, X_embedded_lle, X_embedded_pca, X_embedded_isomap, labels, legend,exp_ind,triats_name=triats_name,filtOrUn=filtOrUn)
    # Display some of the data
    #show_example( eeg_data,'',triats_name=triats_name,filtOrUnfilt=filtOrUn)

    # If we want to visualize in frequency domain/time domain
    # if domain=='Time':
    #     X_embedded=np.fft.ifftshift(fftpack.ifft((eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME, :]),
    #                         axes=1))
    #Plot the data visualization
    # fig = plt.figure()
    if dim==2 and meth!='all':
        show_embedded(X_embedded,labels,legend)
    elif dim==3 and meth!='all':
        fig = plt.figure()
        show_3D(X_embedded,labels,fig,meth,domain)
    else:
        if meth!='all':
            print("Please set dimension to 2/3 for visualization")

def plotErrorMeanAndStdPerExp(pred_mean,pred_std,eigvals_final,svm_mean, svm_std,exp_ind,svm=1):
    '''
    plotErrorMeanAndStdPerExp - Visualizations
    Description - Plot per experiment the pred mean and marginal std and eigenvalues as a function of the dimenstion.
    :param pred_mean[exp_ind][dim_ind]:
    :param pred_std[exp_ind][dim_ind]:
    :param eigvals_final[exp_ind]:
    :param svm_mean[exp_ind]
    :param svm_std[exp_ind]
    :param exp_ind: input for range
    :return: Na
    '''
    fig=[]
    axs=[]
    for i in range(0,exp_ind):
        plt.figure(i)
        plt.suptitle('Experiment '+ str(i))
        plt.subplot(211)
        axes = plt.gca()
        axes.set_ylim([0.5, 1])
        plt.errorbar(range(DIM_RNG_LOW,DIM_RNG_HIGH), np.asarray(pred_mean[i][:]), yerr=np.asarray(pred_std[i][:]), fmt='o', color='b', ecolor='navy')
        plt.plot(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(pred_mean[i][:]), color='black')
        if svm==1:
            plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH], [svm_mean[i],svm_mean[i]], color='red')
            plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH], [svm_mean[i]+svm_std[i],svm_mean[i]+svm_std[i]], color='firebrick')
            plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH], [svm_mean[i]-svm_std[i], svm_mean[i]-svm_std[i]], color='firebrick')
            SVM_patch = mpatches.Patch(color='red', label='SVM score')
            DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
            plt.legend(handles=[DM_patch, SVM_patch])
        else:
            DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
            plt.legend(handles=[DM_patch])
        plt.title('Prediction score as a function of dimension')
        plt.subplot(212)
        plt.scatter(range(1, DIM_RNG_HIGH), np.asarray(eigvals_final[i][-1]))
        plt.title('Eigen values as a function of dimension')
        #plt.show()

def addSVM(svm_mean, svm_std):
    '''
    addSVM - Visualizations
    Description: adds SVM plot to an exsisting plot
    :param svm_mean: value of SVM prediction score
    :param svm_std: value of svm prediction stabdard diviation
    :return: Na
    '''
    plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH-1], [svm_mean, svm_mean], color='red')
    plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH-1], [svm_mean + svm_std, svm_mean + svm_std],
             color='firebrick')
    plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH-1], [svm_mean - svm_std, svm_mean - svm_std],
             color='firebrick')

def plotErrorMeanAndStdPerExpAll(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt,exp_ind,triats_names,svm=1):
    '''
    plotErrorMeanAndStdPerExpAll - Visualizations
    Description - Plot per experiment the pred mean and marginal std and eigenvalues as a function of the dimenstion.
                    including each configuration poaaible under the new data structure
    :param exp_mean_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_std_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_eigvals_unfilt[exp_ind][dim_ind][epsilon_ind]:
    :param svm_mean_unfilt[exp_ind]:
    :param svm_std_unfilt[exp_ind]:
    :param exp_mean_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_std_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_eigvals_filt[exp_ind][dim_ind][epsilon_ind]:
    :param svm_mean_filt[exp_ind]:
    :param svm_std_filt[exp_ind]:
    :param exp_ind: experiments_number
    :param svm: is there an SVM classification for the data i.e not 3 modes labels
    :return: Na
    '''
    #Calculate means scores.
    mean_exp_mean_filt, mean_exp_std_filt, mean_exp_eigvals_filt, mean_svm_mean_filt, mean_svm_std_filt, \
    mean_exp_mean_unfilt, mean_exp_std_unfilt, mean_exp_eigvals_unfilt, mean_svm_mean_unfilt, mean_svm_std_unfilt \
        = calcMeans(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt)

    #Setting variables for max score
    max_score_filt, max_exp_filt, max_dim_filt, max_eps_filt, max_neigh_filt, \
    max_score_unfilt, max_exp_unfilt, max_dim_unfilt, max_eps_unfilt, max_neigh_unfilt, max_filtOrUn = \
        maxScoreExperiments(exp_mean_unfilt, exp_mean_filt)

    # Setting variables for max score
    max_mean_score_filt, max_mean_dim_filt, max_mean_eps_filt, max_mean_neigh_filt, \
    max_mean_score_unfilt, max_mean_dim_unfilt, max_mean_eps_unfilt, max_mean_neigh_unfilt, max_mean_filtOrUn = \
        maxScoreExperimentMeans(mean_exp_mean_unfilt, mean_exp_mean_filt)

    #Display individual configurations
    dim_range = range(DIM_RNG_HIGH - DIM_RNG_LOW)
    for i in range(0,N_experiments):
        triat_name=triats_names[i]
        for k in range(N_epsilon_parameter):
            for l in range(N_neibours_parameter):
                plt.figure(i*N_epsilon_parameter*N_neibours_parameter+k*N_neibours_parameter+l,figsize=(16,9))
                config_string=' - Config: Eps='+str(epsilon_array[k])+' NNeighbors='+str(knn_neibours_array[l])
                file_config_string = '_Config_Eps_'+str(epsilon_array[k])+'_NNeighbors_'+str(knn_neibours_array[l])
                exp_string = 'Experiment '+ str(i)+' '+triat_name
                file_exp_string = 'Experiment_' + str(i)+' '+triat_name
                plt.suptitle(exp_string+config_string)

                plt.subplot(221)
                axes = plt.gca()
                axes.set_ylim([0.5, 1])
                plt.errorbar(range(DIM_RNG_LOW,DIM_RNG_HIGH),
                             [exp_mean_filt[i][x][k][l] for x in dim_range],
                             yerr=[exp_std_filt[i][x][k][l] for x in dim_range],
                             fmt='o', color='b', ecolor='navy')
                for xy in zip(range(DIM_RNG_LOW,DIM_RNG_HIGH), [exp_mean_filt[i][x][k][l] for x in dim_range]):  # <--
                    axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
                plt.plot(range(DIM_RNG_LOW, DIM_RNG_HIGH), [exp_mean_filt[i][x][k][l] for x in dim_range], color='black')
                if svm==1:
                    addSVM(svm_mean_filt[i], svm_std_filt[i])
                    SVM_patch = mpatches.Patch(color='red', label='SVM score - '+'(%.4f)' %svm_mean_filt[i] )
                    DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                    plt.legend(handles=[DM_patch, SVM_patch])
                else:
                    DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                    plt.legend(handles=[DM_patch])
                title221='Prediction score as a function of dimension - ' + ANALYSIS_METHOD_STRING + ' Filtered'
                title223='Eigen values as a function of dimension - '+ ANALYSIS_METHOD_STRING + ' Filtered'
                if (max_exp_filt==i and max_eps_filt==k and max_neigh_filt==l):
                    title221 = title221 + ' - Max score filtered'
                    title223 = title223 + ' - Max score filtered'
                    if (max_filtOrUn==1):
                        title221 = title221 + ' and overall'
                        title223 = title223 + ' and overall'
                plt.title(title221)

                plt.subplot(223)
                axes = plt.gca()
                plt.scatter(range(1, DIM_RNG_HIGH), np.asarray(exp_eigvals_filt[i][-1][k]))
                for xy in zip(range(1,DIM_RNG_HIGH), np.asarray(exp_eigvals_filt[i][-1][k])):  # <--
                    axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
                plt.title(title223)

                plt.subplot(222)
                axes = plt.gca()
                axes.set_ylim([0.5, 1])
                plt.errorbar(range(DIM_RNG_LOW, DIM_RNG_HIGH), [exp_mean_unfilt[i][x][k][l] for x in dim_range],
                             yerr=[exp_std_unfilt[i][x][k][l] for x in dim_range], fmt='o', color='b', ecolor='navy')
                for xy in zip(range(DIM_RNG_LOW, DIM_RNG_HIGH), [exp_mean_unfilt[i][x][k][l] for x in dim_range]):  # <--
                    axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
                plt.plot(range(DIM_RNG_LOW, DIM_RNG_HIGH), [exp_mean_unfilt[i][x][k][l] for x in dim_range], color='black')
                if svm == 1:
                    addSVM(svm_mean_unfilt[i], svm_std_unfilt[i])
                    SVM_patch = mpatches.Patch(color='red', label='SVM score - '+'(%.4f)' %svm_mean_unfilt[i])
                    DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                    plt.legend(handles=[DM_patch, SVM_patch])
                else:
                    DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                    plt.legend(handles=[DM_patch])

                title222 = 'Prediction score as a function of dimension - '+ ANALYSIS_METHOD_STRING + ' Unfiltered'
                title224 = 'Eigen values as a function of dimension - '+ ANALYSIS_METHOD_STRING + ' Unfiltered'
                if (max_exp_unfilt==i and max_eps_unfilt==k and max_neigh_unfilt==l):
                    title222 = title222 + ' - Max score unfiltered'
                    title224 = title224 + ' - Max score unfiltered'
                    if (max_filtOrUn==0):
                        title222 = title222 + ' and overall'
                        title224 = title224 + ' and overall'
                plt.title(title222)

                plt.subplot(224)
                axes = plt.gca()
                plt.scatter(range(1, DIM_RNG_HIGH), np.asarray(exp_eigvals_unfilt[i][-1][k]))
                for xy in zip(range(1,DIM_RNG_HIGH), np.asarray(exp_eigvals_unfilt[i][-1][k])):  # <--
                    axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
                plt.title(title224)
                #plt.show()
                if (SAVE_FIGS == 1):
                    plt.savefig(os.path.join(PATH_RESULTS_DIR,triats_names[i]+'_'+ANALYSIS_METHOD_STRING+'_results', 'Classification_graphs',file_exp_string+file_config_string+'_MeanScoreAndStd.png'))
                plt.close()

    #Display means
    for k in range(N_epsilon_parameter):
        for l in range(N_neibours_parameter):
            plt.figure( k * N_neibours_parameter + l,figsize=(16,9))
            config_string = ' - Config: Eps=' + str(epsilon_array[k]) + ' NNeighbors=' + str(knn_neibours_array[l])
            file_config_string = '_Config_Eps_' + str(epsilon_array[k]) + '_NNeighbors_' + str(knn_neibours_array[l])
            exp_string = 'Experiments Mean'
            file_exp_string = 'Experiments_Mean'
            plt.suptitle(exp_string + config_string)

            plt.subplot(221)
            axes = plt.gca()
            axes.set_ylim([0.5, 1])
            x = []
            for i in range(0,N_dim_parameter):
                x.append(mean_exp_mean_filt[i][k][l][0])
            y = []
            for i in range(0,N_dim_parameter):
                y.append(mean_exp_std_filt[i][k][l][0])
            plt.errorbar(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x),
                         yerr=np.asarray(y), fmt='o', color='b', ecolor='navy')
            for xy in zip(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x)):  # <--
                axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
            plt.plot(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x), color='black')
            if svm == 1:
                addSVM(mean_svm_mean_filt, mean_svm_std_filt)
                SVM_patch = mpatches.Patch(color='red', label='SVM score - '+'(%.4f)' %mean_svm_mean_filt)
                DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                plt.legend(handles=[DM_patch, SVM_patch])
            else:
                DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                plt.legend(handles=[DM_patch])
            title221 = 'Prediction score as a function of dimension - '+ ANALYSIS_METHOD_STRING + ' Filtered '
            title223 = 'Eigen values as a function of dimension - '+ ANALYSIS_METHOD_STRING + ' Filtered'
            if ( max_mean_eps_filt == k and max_mean_neigh_filt == l):
                title221 = title221 + ' - Max score'
                title223 = title223 + ' - Max score'
                if (max_mean_filtOrUn==1):
                    title221 = title221 + ' and overall configurations'
                    title223 = title223 + ' and overall configurations'
            plt.title(title221)

            plt.subplot(223)
            axes = plt.gca()
            plt.scatter(range(1, DIM_RNG_HIGH), np.asarray(mean_exp_eigvals_filt[-1][k]))
            for xy in zip(range(1, DIM_RNG_HIGH), np.asarray(mean_exp_eigvals_filt[-1][k])):  # <--
                axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
            plt.title(title223)

            plt.subplot(222)
            axes = plt.gca()
            axes.set_ylim([0.5, 1])
            x = []
            for i in range(0,N_dim_parameter):
                x.append(mean_exp_mean_unfilt[i][k][l][0])
            y = []
            for i in range(0,N_dim_parameter):
                y.append(mean_exp_std_unfilt[i][k][l][0])
            plt.errorbar(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x),
                         yerr=np.asarray(y), fmt='o', color='b', ecolor='navy')
            for xy in zip(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x)):  # <--
                axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
            plt.plot(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x), color='black')
            if svm == 1:
                addSVM(mean_svm_mean_unfilt, mean_svm_std_unfilt)
                SVM_patch = mpatches.Patch(color='red', label='SVM score - '+'(%.4f)' %mean_svm_mean_unfilt)
                DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                plt.legend(handles=[DM_patch, SVM_patch])
            else:
                DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                plt.legend(handles=[DM_patch])
            title222 = 'Prediction score as a function of dimension - '+ ANALYSIS_METHOD_STRING + ' Unfiltered '
            title224 = 'Eigen values as a function of dimension - '+ ANALYSIS_METHOD_STRING + ' Unfiltered'
            if (max_mean_eps_unfilt == k and max_mean_neigh_unfilt == l ):
                title222 = title222 + ' - Max score'
                title224 = title224 + ' - Max score'
                if (max_mean_filtOrUn == 0):
                    title222 = title222 + ' and overall configurations'
                    title224 = title224 + ' and overall configurations'

            plt.title(title222)

            plt.subplot(224)
            axes = plt.gca()
            plt.scatter(range(1, DIM_RNG_HIGH), np.asarray(mean_exp_eigvals_unfilt[-1][k]))
            for xy in zip(range(1, DIM_RNG_HIGH), np.asarray(mean_exp_eigvals_unfilt[-1][k])):  # <--
                axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
            plt.title(title224)

           # plt.show()
            if (SAVE_FIGS == 1):
                plt.savefig(os.path.join(PATH_RESULTS_DIR, 'Mean_graphs', file_exp_string + file_config_string + '_MeanScoreAndStd_Means.png'))
            plt.close()


# ===== Logs ===== #

def PrintLogToFiles(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt,triats_names,svm=1):
    '''
    PrintLogToFiles - Logs
    Description - Write Log of the data structure per experiment and analysis configuration as function of dimension to a .txt file
    :param exp_mean_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_std_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_eigvals_unfilt[exp_ind][dim_ind][epsilon_ind]:
    :param svm_mean_unfilt[exp_ind]:
    :param svm_std_unfilt[exp_ind]:
    :param exp_mean_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_std_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_eigvals_filt[exp_ind][dim_ind][epsilon_ind]:
    :param svm_mean_filt[exp_ind]:
    :param svm_std_filt[exp_ind]:
    :param svm: 1 if there is svm analysis of experiments, 0 if 3 modes and no SVM evaluation score
    :return: Na
    '''

    # Write results to files
    f_seprate_analysis = open(os.path.join(PATH_RESULTS_DIR, SEPRATE_ANALYSIS_OUTPUT_FILE), 'w')
    for i in range(N_experiments):
        triat_name=triats_names[i]
        for k in range(N_epsilon_parameter):
            for l in range(N_neibours_parameter):
                f_seprate_analysis.write("Experiment - " + str(i + 1) +" "+triat_name+ " - Config: "+ANALYSIS_METHOD_STRING+" Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Unfiltered\n")
                for j in range(N_dim_parameter):
                    f_seprate_analysis.write("Diffusion Dimension - " + str(j + 1) + "\n")
                    f_seprate_analysis.write("Accuracy mean=" + str(exp_mean_unfilt[i][j][k][l]) + "   STD=" + str(exp_std_unfilt[i][j][k][l]) + "\n")
                    f_seprate_analysis.write("Eigan Values - " + "\n")
                    f_seprate_analysis.write(str(exp_eigvals_unfilt[i][j][k]) + "\n")
                    f_seprate_analysis.write("-----------------------------" + "\n")
                if SVM == 1:
                    f_seprate_analysis.write("SVM baseline prediction rate- " + str(svm_mean_unfilt[i]) + "\n")
                f_seprate_analysis.write("==========================================================" + "\n")
                f_seprate_analysis.write("Experiment - " + str(i + 1) +" "+triat_name+" - Config: "+ANALYSIS_METHOD_STRING+" Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Filtered" + "\n")
                for j in range(N_dim_parameter):
                    f_seprate_analysis.write("Diffusion Dimension - " + str(j + 1) + "\n")
                    f_seprate_analysis.write("Accuracy mean=" + str(exp_mean_filt[i][j][k][l]) + "   STD=" + str(exp_std_filt[i][j][k][l]) + "\n")
                    f_seprate_analysis.write("Eigan Values - " + "\n")
                    f_seprate_analysis.write(str(exp_eigvals_filt[i][j][k]) + "\n")
                    f_seprate_analysis.write("-----------------------------" + "\n")
                if SVM == 1:
                    f_seprate_analysis.write("SVM baseline prediction rate- " + str(svm_mean_filt[i]) + "\n")
                f_seprate_analysis.write("==========================================================" + "\n")
    f_seprate_analysis.close()
    # Calculate means and write results to files
    mean_exp_mean_filt, mean_exp_std_filt, mean_exp_eigvals_filt, mean_svm_mean_filt, mean_svm_std_filt, \
    mean_exp_mean_unfilt, mean_exp_std_unfilt, mean_exp_eigvals_unfilt, mean_svm_mean_unfilt, mean_svm_std_unfilt \
        = calcMeans(exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt, svm_mean_unfilt, svm_std_unfilt, exp_mean_filt,
                    exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt)

    # Write results to files
    f_means_analysis = open(os.path.join(PATH_RESULTS_DIR, MEANS_ANALYSIS_OUTPUT_FILE), 'w')

    for k in range(N_epsilon_parameter):
        for l in range(N_neibours_parameter):
            f_means_analysis.write("Experiments means - "+"Config: "+ANALYSIS_METHOD_STRING+" Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Unfiltered" + "\n")
            for j in range(N_dim_parameter):
                f_means_analysis.write("Diffusion Dimension - " + str(j + 1) + "\n")
                f_means_analysis.write("Accuracy mean=" + str(mean_exp_mean_unfilt[j][k][l][0]) + "   STD=" + str(mean_exp_std_unfilt[j][k][l][0]) + "\n")
                f_means_analysis.write("Eigan Values - " + "\n")
                f_means_analysis.write(str(mean_exp_eigvals_unfilt[j][k]) + "\n")
                f_means_analysis.write("-----------------------------" + "\n")
            if SVM == 1:
                f_means_analysis.write("SVM baseline prediction rate- " + str(mean_svm_mean_unfilt) + "\n")
            f_means_analysis.write("==========================================================" + "\n")
            f_means_analysis.write("Experiments means - "+"Config: "+ANALYSIS_METHOD_STRING+" Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Filtered" + "\n")
            for j in range(N_dim_parameter):
                f_means_analysis.write("Diffusion Dimension - " + str(j + 1) + "\n")
                f_means_analysis.write("Accuracy mean=" + str(mean_exp_mean_filt[j][k][l][0]) + "   STD=" + str(mean_exp_std_filt[j][k][l][0]) + "\n")
                f_means_analysis.write("Eigan Values - " + "\n")
                f_means_analysis.write(str(mean_exp_eigvals_filt[j][k]) + "\n")
                f_means_analysis.write("-----------------------------" + "\n")
            if SVM == 1:
                f_means_analysis.write("SVM baseline prediction rate- " + str(mean_svm_mean_filt) + "\n")
            f_means_analysis.write("==========================================================" + "\n")
    f_means_analysis.close()

def PrintLogToScreen(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt,triats_names,svm=1):
    '''
        PrintLogToScreen - Logs
        Description - Write Log of the data structure per experiment and analysis configuration as function of dimension to the console screen
        :param exp_mean_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
        :param exp_std_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
        :param exp_eigvals_unfilt[exp_ind][dim_ind][epsilon_ind]:
        :param svm_mean_unfilt[exp_ind]:
        :param svm_std_unfilt[exp_ind]:
        :param exp_mean_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
        :param exp_std_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
        :param exp_eigvals_filt[exp_ind][dim_ind][epsilon_ind]:
        :param svm_mean_filt[exp_ind]:
        :param svm_std_filt[exp_ind]:
        :param svm: 1 if there is svm analysis of experiments, 0 if 3 modes and no SVM evaluation score
        :return: Na
    '''
    # First try printing results without writing to file
    for i in range(N_experiments):
        triat_name=triats_names[i]
        for k in range(N_epsilon_parameter):
            for l in range(N_neibours_parameter):
                print("Experiment - " + str(i + 1) +" "+triat_name+ " - Config: "+ANALYSIS_METHOD_STRING+" Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Unfiltered")
                for j in range(N_dim_parameter):
                    print("Diffusion Dimension - " + str(j + 1))
                    print("Accuracy mean=" + str(exp_mean_unfilt[i][j][k][l]) + "   STD=" + str(exp_std_unfilt[i][j][k][l]))
                    print("Eigan Values - ")
                    print(str(exp_eigvals_unfilt[i][j][k]))
                    print("-----------------------------")
                if SVM == 1:
                    print("SVM baseline prediction rate- " + str(svm_mean_unfilt[i]))
                print("==========================================================")
                print("Experiment - " + str(i + 1) +" "+triat_name+ " - Config: "+ANALYSIS_METHOD_STRING+" Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Filtered")
                for j in range(N_dim_parameter):
                    print("Diffusion Dimension - " + str(j + 1))
                    print("Accuracy mean=" + str(exp_mean_filt[i][j][k][l]) + "   STD=" + str(exp_std_filt[i][j][k][l]))
                    print("Eigan Values - ")
                    print(str(exp_eigvals_filt[i][j][k]))
                    print("-----------------------------")
                if SVM == 1:
                    print("SVM baseline prediction rate- " + str(svm_mean_filt[i]))
                print("==========================================================")

    mean_exp_mean_filt, mean_exp_std_filt, mean_exp_eigvals_filt, mean_svm_mean_filt, mean_svm_std_filt, \
    mean_exp_mean_unfilt, mean_exp_std_unfilt, mean_exp_eigvals_unfilt, mean_svm_mean_unfilt, mean_svm_std_unfilt \
        = calcMeans(exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt, svm_mean_unfilt, svm_std_unfilt, exp_mean_filt,
                    exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt)
    # Try first to print before file printing
    for k in range(N_epsilon_parameter):
        for l in range(N_neibours_parameter):
            print("Experiments means - "+"Config: "+ANALYSIS_METHOD_STRING+" Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Unfiltered")
            for j in range(N_dim_parameter):
                print("Diffusion Dimension - " + str(j + 1))
                print("Accuracy mean=" + str(mean_exp_mean_unfilt[j][k][l][0]) + "   STD=" + str(mean_exp_std_unfilt[j][k][l][0]))
                print("Eigan Values - ")
                print(str(mean_exp_eigvals_unfilt[j][k]))
                print("-----------------------------")
            if SVM == 1:
                print("SVM baseline prediction rate- " + str(mean_svm_mean_unfilt))
            print("==========================================================")
            print("Experiments means - "+"Config: "+ANALYSIS_METHOD_STRING+" Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Filtered")
            for j in range(N_dim_parameter):
                print("Diffusion Dimension - " + str(j + 1))
                print("Accuracy mean=" + str(mean_exp_mean_filt[j][k][l][0]) + "   STD=" + str(mean_exp_std_filt[j][k][l][0]))
                print("Eigan Values - ")
                print(str(mean_exp_eigvals_filt[j][k]))
                print("-----------------------------")
            if SVM == 1:
                print("SVM baseline prediction rate- " + str(mean_svm_mean_filt))
            print("==========================================================")

def PrintConfigToFile(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt,triats_names,svm=1):
    '''
    PrintConfigToFile - Logs
    Description - Write full analysis configuration to a .txt file, including, triats names, max score configurations and and max classification over all triats,
    :param exp_mean_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_std_unfilt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_eigvals_unfilt[exp_ind][dim_ind][epsilon_ind]:
    :param svm_mean_unfilt[exp_ind]:
    :param svm_std_unfilt[exp_ind]:
    :param exp_mean_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_std_filt[exp_ind][dim_ind][epsilon_ind][KNNeibours_ind]:
    :param exp_eigvals_filt[exp_ind][dim_ind][epsilon_ind]:
    :param svm_mean_filt[exp_ind]:
    :param svm_std_filt[exp_ind]:
    :param svm: 1 if there is svm analysis of experiments, 0 if 3 modes and no SVM evaluation score
    :return: Na
    '''
    #Calculate means scores.
    mean_exp_mean_filt, mean_exp_std_filt, mean_exp_eigvals_filt, mean_svm_mean_filt, mean_svm_std_filt, \
    mean_exp_mean_unfilt, mean_exp_std_unfilt, mean_exp_eigvals_unfilt, mean_svm_mean_unfilt, mean_svm_std_unfilt \
        = calcMeans(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt)

    #Setting variables for max score
    max_score_filt, max_exp_filt, max_dim_filt, max_eps_filt, max_neigh_filt, \
    max_score_unfilt, max_exp_unfilt, max_dim_unfilt, max_eps_unfilt, max_neigh_unfilt, max_filtOrUn = \
        maxScoreExperiments(exp_mean_unfilt, exp_mean_filt)

    # Setting variables for max score
    max_mean_score_filt, max_mean_dim_filt, max_mean_eps_filt, max_mean_neigh_filt, \
    max_mean_score_unfilt, max_mean_dim_unfilt, max_mean_eps_unfilt, max_mean_neigh_unfilt, max_mean_filtOrUn = \
        maxScoreExperimentMeans(mean_exp_mean_unfilt, mean_exp_mean_filt)

    # Write results to files
    f_config = open(os.path.join(PATH_RESULTS_DIR, CONFIG_ANALYSIS_OUTPUT_FILE), 'w')
    f_config.write("-----------------------------" + "\n")
    f_config.write("Data: ")
    if (SKIP_BLINK==1):
        f_config.write("		open closed states only." + "\n\n")
    else:
        f_config.write("		open, closed, blink states all." + "\n\n")
    f_config.write("Overlap:" + str(int(OVERLAP*100)) + "%" + "\n")
    f_config.write("-------- Preprocessing - -------" + "\n")
    f_config.write("Filtering: "+"\n")
    f_config.write("Buttersworth," + "\n")
    f_config.write("order: " + str(ORDER)+ ",\n")
    f_config.write("low cut: " + str(LOW_CUT) + " [Hz],"+"\n")
    f_config.write("high cut: " + str(HIGH_CUT) + " [Hz]," + "\n")
    f_config.write("Analysis on:        eeg " +ANALYSIS_METHOD_STRING+ " spectrum unfiltered, eeg " +ANALYSIS_METHOD_STRING+ " filtered" + "\n")
    f_config.write("FFt cut: ["+str(FFT_VECTOR_CUT_LOW)+", "+str(FFT_VECTOR_CUT_HIGH)+"] [Hz]"+ "\n\n")
    f_config.write("-------- Diffusion maps --------" + "\n")
    f_config.write("Epsilon: " + str(EPSILON_LOW) +" - "+str(int(EPSILON_HIGH/10)) +"\n")
    f_config.write("Knn" + "\n")
    f_config.write("Neighnours: " + str(N_NEIGHBORS_LOW) + " - " + str(int(N_NEIGHBORS_HIGH-2))+"\n")
    f_config.write("Dimensions: " + str(DIM_RNG_LOW) + " - " + str(int(DIM_RNG_HIGH - 1)) + "\n\n")
    f_config.write("Patients: " + "\n")
    for i in range(len(triats_names)):
        f_config.write(triats_names[i] + "\n")
    f_config.write("\n"+"-------- Max classification scores --------" + "\n")
    f_config.write("Max classification score unfiltered: " + str(max_score_unfilt) + "\n")
    f_config.write("Max classification config unfiltered:" + "\n")
    f_config.write("Experiment number: " +str(max_exp_unfilt) + " - "+triats_names[max_exp_unfilt]+"\n")
    f_config.write("Epsilon: " +str(epsilon_array[max_eps_unfilt]) + "\n")
    f_config.write("N neibours: " + str(knn_neibours_array[max_neigh_unfilt]) + "\n")
    f_config.write("Dimension: " + str(int(max_dim_unfilt)+int(DIM_RNG_LOW)) + "\n\n")
    f_config.write("Max classification score filtered: " + str(max_score_filt) + "\n")
    f_config.write("Max classification config filtered:" + "\n")
    f_config.write("Experiment number: " + str(max_exp_filt)  + " - "+triats_names[max_exp_filt]+ "\n")
    f_config.write("Epsilon: " + str(epsilon_array[max_eps_filt]) + "\n")
    f_config.write("N neibours: " + str(knn_neibours_array[max_neigh_filt]) + "\n")
    f_config.write("Dimension: " + str(int(max_dim_filt) + int(DIM_RNG_LOW)) + "\n\n")
    f_config.write("---- Means ----" + "\n")
    f_config.write("Max classification score unfiltered: " + str(max_mean_score_unfilt) + "\n")
    f_config.write("Max classification config unfiltered:" + "\n")
    f_config.write("Epsilon: " + str(epsilon_array[max_mean_eps_unfilt]) + "\n")
    f_config.write("N neibours: " + str(knn_neibours_array[max_mean_neigh_unfilt]) + "\n")
    f_config.write("Dimension: "  +str(int(max_mean_dim_unfilt)+int(DIM_RNG_LOW)) + "\n\n")
    f_config.write("Max classification score filtered: " + str(max_mean_score_filt) + "\n")
    f_config.write("Max classification config filtered:" + "\n")
    f_config.write("Epsilon: " + str(epsilon_array[max_mean_eps_filt]) + "\n")
    f_config.write("N neibours: " + str(knn_neibours_array[max_mean_neigh_filt]) + "\n")
    f_config.write("Dimension: "  +str(int(max_mean_dim_filt) + int(DIM_RNG_LOW)) + "\n\n")
    f_config.close()


################################### Main ######################################
if __name__ == '__main__':
    # ----------------------------------------------------- #
    # --- Part A - loading saved data (working offline) --- #
    # ----------------------------------------------------- #
    file_names = os.listdir(RECORDS_PATH)
    full_paths = [os.path.join(RECORDS_PATH, fn) for fn in file_names if 'configuration' not in fn]

    if not os.path.exists(PATH_RESULTS_DIR):
        os.mkdir(PATH_RESULTS_DIR)

    # ======= Data structure preperation ========= #
    N_experiments = len(full_paths)
    # creating data structure to accomadate the classification results
    exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt = [[[[[] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)] , [[[[[] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)] , [[[[] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)]
    exp_mean_filt, exp_std_filt, exp_eigvals_filt = [[[[[] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)] , [[[[[] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)] , [[[[] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)]
    svm_mean_unfilt = {}
    svm_std_unfilt = {}
    svm_mean_filt = {}
    svm_std_filt = {}
    triats_names = ['' for i in range(N_experiments)]

    variables_path = os.path.join(PATH_RESULTS_DIR, 'variables_' + str(EXPERIMENT_TYPE) + '.txt')
    if not os.path.exists(os.path.join(PATH_RESULTS_DIR, 'Mean_graphs')):
        os.mkdir(os.path.join(PATH_RESULTS_DIR, 'Mean_graphs'))

    is_igor = False
    if not os.path.exists(variables_path):
        for exp_ind, data_path in enumerate(full_paths):  # for each experiment
            triats_names[exp_ind] = get_triat_name(exp_ind,file_names[exp_ind])

            # set special channels for igor triats experiment
            if triats_names[exp_ind] == 'Igor':
                is_igor = True
                TEMP_CHANNELS = ACTIVE_CHANNELS
                ACTIVE_CHANNELS = IGOR_CHANNELS
            elif is_igor:
                ACTIVE_CHANNELS = TEMP_CHANNELS

            # Creation of relevant results directories if needed
            triat_path = os.path.join(PATH_RESULTS_DIR, triats_names[exp_ind] +'_'+ANALYSIS_METHOD_STRING+ '_results')
            CUR_TRIAT_PATH=triat_path
            if not os.path.exists(triat_path):
                os.mkdir(triat_path)
            if not os.path.exists(os.path.join(triat_path, 'Visualizations')):
                os.mkdir(os.path.join(triat_path, 'Visualizations'))
            if not os.path.exists(os.path.join(triat_path, 'Examples_time')):
                os.mkdir(os.path.join(triat_path, 'Examples_time'))
            if not os.path.exists(os.path.join(triat_path, 'Examples_frequency')):
                os.mkdir(os.path.join(triat_path, 'Examples_frequency'))
            if not os.path.exists(os.path.join(triat_path, 'Examples_averages')):
                os.mkdir(os.path.join(triat_path, 'Examples_averages'))
            if not os.path.exists(os.path.join(triat_path, 'Diffusion_results')):
                os.mkdir(os.path.join(triat_path, 'Diffusion_results'))
            if not os.path.exists(os.path.join(triat_path, 'Classification_graphs')):
                os.mkdir(os.path.join(triat_path, 'Classification_graphs'))

            # load and filter the data according to the frequency analysis method and axtract the time frames segments as well
            if(FREQUENCY_ANALYSIS_METHOD=='fft'):
                if (exp_ind==0):
                    eeg_fft_unfilt, eeg_fft_filt = load_and_filter_data(data_path,plotPlease=True,meth=FREQUENCY_ANALYSIS_METHOD)
                    eeg_segments = load_and_filter_data(data_path,filter=False, plotPlease=False,meth=FREQUENCY_ANALYSIS_METHOD)
                else:
                    eeg_fft_unfilt, eeg_fft_filt = load_and_filter_data(data_path, plotPlease=False,meth=FREQUENCY_ANALYSIS_METHOD)
                    eeg_segments = load_and_filter_data(data_path, filter=False, plotPlease=False,meth=FREQUENCY_ANALYSIS_METHOD)
            if (FREQUENCY_ANALYSIS_METHOD == 'welch'):
                if (exp_ind == 0):
                    f_ax, eeg_welch_unfilt, eeg_welch_filt = load_and_filter_data(data_path, plotPlease=True,
                                                                        meth=FREQUENCY_ANALYSIS_METHOD)
                    eeg_segments = load_and_filter_data(data_path, filter=False, plotPlease=False,
                                                        meth=FREQUENCY_ANALYSIS_METHOD)
                else:
                    f_ax, eeg_welch_unfilt, eeg_welch_filt = load_and_filter_data(data_path, plotPlease=False,
                                                                        meth=FREQUENCY_ANALYSIS_METHOD)
                    eeg_segments = load_and_filter_data(data_path, filter=False, plotPlease=False,
                                                        meth=FREQUENCY_ANALYSIS_METHOD)

            # ---- Examples reporting - Individual time frames ---- #

            show_example_freq_avg(eeg_segments, 'Experiment ' + str(exp_ind) + ' - ' + triats_names[exp_ind] + ' Frequncy domain examples - Unfiltered', triats_name=triats_names[exp_ind], filtOrUnfilt=0,half_win=5)
            show_example_time(eeg_segments, triats_names[exp_ind],filtOrUnfilt=0)
            if (FREQUENCY_ANALYSIS_METHOD == 'fft'):
                show_example_freq(eeg_fft_unfilt, 'Experiment '+str(exp_ind)+' - '+ triats_names[exp_ind]+' Frequncy domain examples - Unfiltered', triats_name=triats_names[exp_ind], filtOrUnfilt=0)
                show_example_freq(eeg_fft_filt, 'Experiment '+str(exp_ind)+' - '+ triats_names[exp_ind]+' Frequncy domain examples - Filtered', triats_name=triats_names[exp_ind], filtOrUnfilt=1)
                eeg_preprocessed_unfilt = preprocess_data(eeg_fft_unfilt, [FFT_VECTOR_CUT_LOW,FFT_VECTOR_CUT_HIGH])  # hardcoded frequency cut
                eeg_preprocessed_filt = preprocess_data(eeg_fft_filt, [FFT_VECTOR_CUT_LOW,  FFT_VECTOR_CUT_HIGH])    # hardcoded frequency cut
            elif (FREQUENCY_ANALYSIS_METHOD == 'welch'):
                show_example_freq_welch(eeg_welch_unfilt, 'Experiment ' + str(exp_ind) + ' - ' + triats_names[exp_ind] + ' Frequncy domain examples - Unfiltered', triats_name=triats_names[exp_ind],filtOrUnfilt=0)
                show_example_freq_welch(eeg_welch_filt, 'Experiment ' + str(exp_ind) + ' - ' + triats_names[exp_ind] + ' Frequncy domain examples - Filtered', triats_name=triats_names[exp_ind],filtOrUnfilt=1)
                eeg_preprocessed_unfilt = preprocess_data(eeg_welch_unfilt, [FFT_VECTOR_CUT_LOW, FFT_VECTOR_CUT_HIGH])  # hardcoded frequency cut
                eeg_preprocessed_filt = preprocess_data(eeg_welch_filt, [FFT_VECTOR_CUT_LOW,FFT_VECTOR_CUT_HIGH])  # hardcoded frequency cut

            # =====Changes for a different experiment configuration ===== #
            labels, legend = get_aviv_exp_timeframes(eeg_preprocessed_unfilt)
            if (FREQUENCY_ANALYSIS_METHOD=='fft'):
                eeg_flatten_unfilt = np.asarray([x.flatten() for x in eeg_preprocessed_unfilt])
                eeg_flatten_filt = np.asarray([x.flatten() for x in eeg_preprocessed_filt])
            elif (FREQUENCY_ANALYSIS_METHOD=='welch'):
                dim_1=len(eeg_preprocessed_unfilt)
                dim_2=len((np.asarray([x.flatten() for x in eeg_preprocessed_unfilt[0]])).flatten())
                eeg_flatten_unfilt_l=[[0 for x in range(dim_2)] for y in range(dim_1)]
                eeg_flatten_filt_l = [[0 for x in range(dim_2)] for y in range(dim_1)]
                for i in range(len(eeg_preprocessed_unfilt)):
                    eeg_flatten_unfilt_l[i][:] = (np.asarray([x.flatten()for x in eeg_preprocessed_unfilt[i]])).flatten()
                    eeg_flatten_filt_l[i][:] = (np.asarray([x.flatten() for x in eeg_preprocessed_filt[i]])).flatten()
                # ==== Using only Open / Close data ==== #
                eeg_flatten_unfilt = np.asarray(eeg_flatten_unfilt_l)
                eeg_flatten_filt = np.asarray(eeg_flatten_filt_l)

            # if Experiment 1 and we want to analize to states only extract states
            if SKIP_BLINK == 1 and EXPERIMENT_TYPE==1:
                data_0_unfilt, labels_0 = extract_state(eeg_flatten_unfilt, labels, 0)
                data_2_unfilt, labels_2 = extract_state(eeg_flatten_unfilt, labels, 2)
                data_0_filt, labels_0 = extract_state(eeg_flatten_filt, labels, 0)
                data_2_filt, labels_2 = extract_state(eeg_flatten_filt, labels, 2)
                data_02_filt = data_0_filt + data_2_filt
                data_02_unfilt = data_0_unfilt + data_2_unfilt
                labels_02 = labels_0 + labels_2
            elif EXPERIMENT_TYPE==2 or EXPERIMENT_TYPE==3:
                data_02_filt = eeg_flatten_filt
                data_02_unfilt = eeg_flatten_unfilt
                labels_02 = labels

            # other embedding visualization methods
            if SKIP_BLINK == 1:
                visualize_eeg(data_02_unfilt[:-1],labels_02[:-1], legend, FS, domain='Freq', meth='all', dim=2, exp_ind=exp_ind,triats_name=triats_names[exp_ind], filtOrUn=0)
                visualize_eeg(data_02_filt[:-1], labels_02[:-1], legend, FS, domain='Freq', meth='all', dim=2, exp_ind=exp_ind,triats_name=triats_names[exp_ind], filtOrUn=1)
                do_nothing = 5
            else:
                visualize_eeg(eeg_flatten_unfilt[:-1], labels[:-1], legend, FS, domain='Freq', meth='all', dim=2,exp_ind=exp_ind, filtOrUn=0)
                visualize_eeg(eeg_flatten_filt[:-1], labels[:-1], legend, FS, domain='Freq', meth='all', dim=2, exp_ind=exp_ind, filtOrUn=1)
            # ====================================== #

            # ======= Diffusion maps for data ========= #
            for epsilon_index in range(N_epsilon_parameter):
                epsilon = epsilon_array[epsilon_index]  # diffusion distance epsilon
                dim_index=0
                for dimension in range(DIM_RNG_LOW, DIM_RNG_HIGH):
                    coords_out_unfilt, labels_out_unfilt = [], []
                    coords_out_filt, labels_out_filt = [], []
                    if SKIP_BLINK == 1:
                        coords_unfilt, dataList, eigvals_unfilt = dm.diffusionMapping(data_02_unfilt[:-1], lambda x, y: math.exp(-LA.norm(x - y) / epsilon), t=2, dim=dimension)
                        labels_out_unfilt.append(labels_02[:-1])
                        coords_filt, dataList, eigvals_filt = dm.diffusionMapping(data_02_filt[:-1], lambda x, y: math.exp(-LA.norm(x - y) / epsilon), t=2, dim=dimension)
                        labels_out_filt.append(labels_02[:-1])
                        if (dimension==DIM_RNG_HIGH-1):
                            [diffusion_config_string, file_diffusion_config_string]=get_config_diffusion(exp_ind=exp_ind, eps_ind=epsilon_index,triats_name=triats_names[exp_ind])
                            show_diffusion(coords_unfilt, labels_out_unfilt, legend,config_string=file_diffusion_config_string, filtOrUnfilt=0)
                            show_diffusion(coords_filt, labels_out_filt, legend,config_string=file_diffusion_config_string,filtOrUnfilt=1)
                    else:
                        coords_unfilt, dataList, eigvals_unfilt = dm.diffusionMapping(eeg_flatten_unfilt[:-1], lambda x, y: math.exp(-LA.norm(x - y) / epsilon), t=2, dim=dimension)
                        labels_out_unfilt.append(labels[:-1])
                        coords_filt, dataList, eigvals_filt = dm.diffusionMapping(eeg_flatten_filt[:-1], lambda x, y: math.exp(-LA.norm(x - y) / epsilon), t=2, dim=dimension)
                        labels_out_filt.append(labels[:-1])
                        # show_diffusion(coords_unfilt, labels_out_unfilt, legend)
                        # show_diffusion(coords_filt, labels_out_filt, legend)

                    coords_out_unfilt.append(coords_unfilt)
                    coords_out_filt.append(coords_filt)

                    # ===== KNN classification on the embedded coordinates in diffusion space ===== #
                    for neibours_index in range(N_neibours_parameter):
                        pred_mean_unfilt, pred_std_unfilt = knn_clustering(coords_out_unfilt, labels_out_unfilt, knn_neibours_array[neibours_index])
                        exp_mean_unfilt[exp_ind][dim_index][epsilon_index][neibours_index] = pred_mean_unfilt
                        exp_std_unfilt[exp_ind][dim_index][epsilon_index][neibours_index] = pred_std_unfilt

                        pred_mean_filt, pred_std_filt = knn_clustering(coords_out_filt, labels_out_filt, knn_neibours_array[neibours_index])
                        exp_mean_filt[exp_ind][dim_index][epsilon_index][neibours_index] = pred_mean_filt
                        exp_std_filt[exp_ind][dim_index][epsilon_index][neibours_index] = pred_std_filt

                    exp_eigvals_unfilt[exp_ind][dim_index][epsilon_index] = eigvals_unfilt
                    exp_eigvals_filt[exp_ind][dim_index][epsilon_index] = eigvals_filt
                    dim_index = dim_index+1

            # ===== SVM training and classification per experiment ===== #
            if SVM == 1:    ## for each experiment once
                svm_pred_mean_unfilt, svm_pred_std_unfilt = svm_cross_val(data_02_unfilt, labels_02)
                svm_mean_unfilt[exp_ind] = svm_pred_mean_unfilt
                svm_std_unfilt[exp_ind] = svm_pred_std_unfilt
                svm_pred_mean_filt, svm_pred_std_filt = svm_cross_val(data_02_filt, labels_02)
                svm_mean_filt[exp_ind] = svm_pred_mean_filt
                svm_std_filt[exp_ind] = svm_pred_std_filt

            # finished processing note
            print('Finished calculating diffusion maps and SVM for expeiment - ' + str(exp_ind))

        file = open(variables_path, 'wb')
        results_dict = {'mean_unf': exp_mean_unfilt,
                        'mean_f': exp_mean_filt,
                        'std_f': exp_std_filt,
                        'std_unf': exp_std_unfilt,
                        'mean_svm_unf': svm_mean_unfilt,
                        'mean_svm_f': svm_mean_filt,
                        'std_svm_unf': svm_std_unfilt,
                        'std_svm_f': svm_std_filt,
                        'eigvals_f': exp_eigvals_filt,
                        'eigvals_unf': exp_eigvals_unfilt,
                        'triats_names': triats_names
                        }
        pickle.dump(results_dict, file)
        file.close()
    else:
        # if variables file exists in result folder load it and use the prediction data to plot graphs later on in the program
        file = open(variables_path, 'rb')
        results_dict = pickle.load(file)
        exp_mean_unfilt = results_dict['mean_unf']
        exp_mean_filt = results_dict['mean_f']
        exp_std_unfilt = results_dict['std_unf']
        exp_std_filt = results_dict['std_f']
        svm_mean_unfilt = results_dict['mean_svm_unf']
        svm_mean_filt = results_dict['mean_svm_f']
        svm_std_unfilt = results_dict['std_svm_unf']
        svm_std_filt = results_dict['std_svm_f']
        exp_eigvals_filt = results_dict['eigvals_f']
        exp_eigvals_unfilt = results_dict['eigvals_unf']
        triats_names = results_dict['triats_names']
        file.close()

    # # write results to xml
    # write_xml(os.path.join(PATH_RESULTS_DIR, 'results.xlsx'), exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt,
    #           N_experiments, N_epsilon_parameter, N_neibours_parameter, N_dim_parameter,
    #           SVM=True, mean_svm=svm_mean_unfilt, filtered=False)
    # write_xml(os.path.join(PATH_RESULTS_DIR, 'results.xlsx'), exp_mean_filt, exp_std_filt, exp_eigvals_filt,
    #           N_experiments, N_epsilon_parameter, N_neibours_parameter, N_dim_parameter,
    #           SVM=True, mean_svm=svm_mean_filt, filtered=True)
    # ======================================================== #

    # if loading from a analysis variables file and results directories don't exist create them
    if not os.path.exists(PATH_RESULTS_DIR):
        os.mkdir(PATH_RESULTS_DIR)
    if not os.path.exists(os.path.join(PATH_RESULTS_DIR, 'Mean_graphs')):
        os.mkdir(os.path.join(PATH_RESULTS_DIR, 'Mean_graphs'))
    for i in range(len(triats_names)):
        triat_path=os.path.join(PATH_RESULTS_DIR, triats_names[i]+'_'+ANALYSIS_METHOD_STRING+'_results')
        if not os.path.exists(triat_path):
            os.mkdir(triat_path)
        if not os.path.exists(os.path.join(triat_path, 'Visualizations')):
            os.mkdir(os.path.join(triat_path, 'Visualizations'))
        if not os.path.exists(os.path.join(triat_path, 'Examples_time')):
            os.mkdir(os.path.join(triat_path, 'Examples_time'))
        if not os.path.exists(os.path.join(triat_path, 'Examples_frequency')):
            os.mkdir(os.path.join(triat_path, 'Examples_frequency'))
        if not os.path.exists(os.path.join(triat_path, 'Examples_averages')):
            os.mkdir(os.path.join(triat_path, 'Examples_averages'))
        if not os.path.exists(os.path.join(triat_path, 'Diffusion_results')):
            os.mkdir(os.path.join(triat_path, 'Diffusion_results'))
        if not os.path.exists(os.path.join(triat_path, 'Classification_graphs')):
            os.mkdir(os.path.join(triat_path, 'Classification_graphs'))

    # ===== Report results and print them to the screen ===== #
    plotErrorMeanAndStdPerExpAll(exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt, svm_mean_unfilt, svm_std_unfilt,exp_mean_filt, exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt, N_experiments,triats_names=triats_names,svm=SVM)
    PrintConfigToFile(exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt, svm_mean_unfilt, svm_std_unfilt, exp_mean_filt, exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt,triats_names=triats_names, svm=SVM)
    PrintLogToFiles(exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt, svm_mean_unfilt, svm_std_unfilt, exp_mean_filt, exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt,triats_names=triats_names, svm=SVM)
    PrintLogToScreen(exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt, svm_mean_unfilt, svm_std_unfilt, exp_mean_filt, exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt,triats_names=triats_names, svm=SVM)

    ttt= 5


# write to excel the results of the experiments





