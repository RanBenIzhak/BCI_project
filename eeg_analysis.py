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


ACTIVE_CHANNELS = (2, 5, 6, 8)
FS = 250
LOW_CUT = 2.5
HIGH_CUT = 22
ORDER = 8       # up to 8
OVERLAP=0.5     #   0-0.5   ##############
TIME_FRAME= 1      #[Sec]        ##############
SAMPLES_PER_FRAME= TIME_FRAME * FS      #(TIME_FRAME/FS)*SAMPLES_PER_FRAME<=1 ##############
HALF_WIN_AVG = 5
PLOT_EXAMPLE = False
SKIP_BLINK = 1
if SKIP_BLINK == 0:
    SVM = 0
else:
    SVM = 1
# ===== KNN constants ==== #
N_NEIGHBORS_LOW = 3
N_NEIGHBORS_HIGH = 5
n_components = 5 #ALL
n_neighbors = 5  #LLE, Isomap
# ==== Diffusion parameters ==== #
DIM_RNG_LOW = 1
DIM_RNG_HIGH = 4
EPSILON_LOW = 100
EPSILON_HIGH = 1000
# ==== Run Configurations ==== #
SAVE_FIGS = 1
EXPERIMENT_TYPE = 1   # 1 - old ran\aviv's experiment, 2 - new (6 participants) experiment
FIND_MAX = 1
VISUALIZE_DATA = 1

# ==== PATHS AND FILE NAMES ==== #
cur_path = os.path.dirname(os.path.realpath(__file__))
RECORDS_PATH = os.path.join(cur_path, 'Data', 'Experiment ' + str(EXPERIMENT_TYPE))
PATH_RESULTS_DIR = os.path.join(cur_path, 'Results')
SEPRATE_ANALYSIS_OUTPUT_FILE = 'Seprate_analysis_output.txt'
MEANS_ANALYSIS_OUTPUT_FILE = 'Means_analysis_output.txt'


def show_embedded(coordinates, labels, legend):
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
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Method = %s, Domain = %s' % (meth, domain))
        for i in np.sort(np.unique(labels)):
            indices= labels == i
            ax.scatter(np.asarray(X_embedded[indices,0]),np.asarray(X_embedded[indices,1]),np.asarray(X_embedded[indices,2]),label=legend[i])
        plt.show()

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
#Just change here the 20 to 150 to set to new experiments
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
    samples_per_mode = int(20 / (TIME_FRAME * (1 - OVERLAP)))#Aviv : I belive this 20 turns to 150 seconds per mode and we can then ru on the new data
    labels = np.zeros(session_len)
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
    fig = plt.figure()
    a = np.asarray(coordinates)
    labels = np.asarray(labels_list)
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

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def knn_clustering(data, labels, neighbors_num=N_NEIGHBORS_LOW):
    '''
    
    :param data: 
    :param labels: 
    :return: 
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

def extract_state(data, labels, label_to_extract):
    '''
    Extract certain data with certain labels - for ignoring blinking data
    :param data: 
    :param labels: 
    :param label_to_extract: 
    :return: 
    '''
    cur_d = [d1 for d1, l1 in zip(data, labels) if l1 == label_to_extract]
    cur_l = [l1 for l1 in labels if l1 == label_to_extract]
    return cur_d, cur_l

def pair_wise_knn(data, labels, nn=3):
    '''
     0 vs 2 (open vs closed)
    :param data: 
    :param labels: 
    :return: 
    '''
    data_0, labels_0 = extract_state(data, labels, 0)
    # data_1, labels_1 = extract_state(data, labels, 1)
    data_2, labels_2 = extract_state(data, labels, 2)
    data_02 = [d0 + d2 for d0, d2 in zip(data_0, data_2)]
    labels_02 = [l0 + l2 for l0, l2 in zip(labels_0, labels_2)]
    pred_0v2, std_0v2 = knn_clustering(data_02, labels_02, neighbors_num=nn)
    return (pred_0v2, std_0v2)

def preprocess_data(eeg_data, cut_band):
    '''
    
    :param eeg_data: data input (in frequency domain!)
    :param cut_band: 2 values - [freq_low, freq_high]  [Hz]
    :return: eeg_cut: processed data - cutted values (NOT FILTERED) between [ cut_band[0] , cut_band[1] ]
    '''
    num_samples_per_window = FS * TIME_FRAME
    freq_high_per_window = int(FS / 2)
    delta_f = freq_high_per_window / num_samples_per_window
    cut_band_ind_low = int(cut_band[0] / delta_f)
    cut_band_ind_high = int(cut_band[1] / delta_f)
    eeg_cut = [x[cut_band_ind_low:cut_band_ind_high, :] for x in eeg_data]

    return eeg_cut

def show_embedded_all(coordinates_tsne,coordinates_lle,coordinates_pca, coordinates_isomap, labels, legend,exp_ind, filtOrUn=0):
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
    exp_string='Experiment '+str(exp_ind)
    file_name_string='Experiment_'+str(exp_ind)+'_Visualization_'
    if(filtOrUn==0):
        fig_title=fig_title+' - Unfiltered'
        file_name_string=file_name_string+'Unfiltered.png'
    else :
        fig_title = fig_title + ' - Filtered'
        file_name_string = file_name_string + 'Filtered.png'
    plt.suptitle(fig_title)
 #   plt.show()
    if(SAVE_FIGS==1):
        plt.savefig(os.path.join(PATH_RESULTS_DIR,file_name_string))
    plt.close()

def visualize_eeg(eeg_data, labels, legend, fs, domain='Freq', meth='tsne', dim=2, exp_ind=1, filtOrUn=0):
    '''
    Visualize the input eeg data in time domain
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
        show_embedded_all(X_embedded_tsne, X_embedded_lle, X_embedded_pca, X_embedded_isomap, labels, legend,exp_ind, filtOrUn=filtOrUn)

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
    plotErrorMeanAndStdPerExp:
    Description - Plot per experiment the pred mean and marginal std and eigenvalues as a function of the dimenstion.
    :param pred_mean[exp_ind][dim_ind]:
    :param pred_std[exp_ind][dim_ind]:
    :param eigvals_final[exp_ind]:
    :param svm_mean[exp_ind]
    :param svm_std[exp_ind]
    :param exp_ind: input for range
    :return:
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
    plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH-1], [svm_mean, svm_mean], color='red')
    plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH-1], [svm_mean + svm_std, svm_mean + svm_std],
             color='firebrick')
    plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH-1], [svm_mean - svm_std, svm_mean - svm_std],
             color='firebrick')

def plotErrorMeanAndStdPerExpAll(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt,exp_ind,svm=1):
    '''
    plotErrorMeanAndStdPerExpAll:
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
    :return:
    '''
    fig=[]
    axs=[]
    N_epsilon_parameter = int(math.log10(EPSILON_HIGH / EPSILON_LOW))  # requires running new DM
    epsilon_array = np.asarray([EPSILON_LOW * math.pow(10, i) for i in range(int(N_epsilon_parameter))])
    knn_neibours_array = np.asarray(range(N_NEIGHBORS_LOW, N_NEIGHBORS_HIGH, 2))
    N_neibours_parameter = len(knn_neibours_array)  # does not require neq
    N_experiments = len(full_paths)
    N_dim_parameter=DIM_RNG_HIGH-DIM_RNG_LOW
    #Setting variables for average scores calculations.
    mean_exp_mean_unfilt, mean_exp_std_unfilt, mean_exp_eigvals_unfilt = [[[[0.0] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)], [[[[0.0] for i in range( N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)], [[[] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)]
    mean_exp_mean_filt, mean_exp_std_filt, mean_exp_eigvals_filt = [[[[0.0] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)], [[[[0.0] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)], [[[] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)]
    mean_svm_mean_unfilt = 0.0
    mean_svm_std_unfilt = 0.0
    mean_svm_mean_filt = 0.0
    mean_svm_std_filt = 0.0
    for i in range(N_experiments):
        mean_svm_mean_filt =mean_svm_mean_filt+ svm_mean_filt[i]/N_experiments
        mean_svm_std_filt = mean_svm_std_filt + svm_std_filt[i] / N_experiments
        mean_svm_mean_unfilt = mean_svm_mean_unfilt + svm_mean_unfilt[i]/N_experiments
        mean_svm_std_unfilt = mean_svm_std_unfilt + svm_std_unfilt[i] / N_experiments
        for j in range(N_dim_parameter):
            for k in range(N_epsilon_parameter):
                if (i==0):
                    mean_exp_eigvals_filt[j][k] = exp_eigvals_filt[i][j][k]/N_experiments
                    mean_exp_eigvals_unfilt[j][k] = exp_eigvals_unfilt[i][j][k]/N_experiments
                else:
                    mean_exp_eigvals_filt[j][k] = mean_exp_eigvals_filt[j][k] + exp_eigvals_filt[i][j][k] / N_experiments
                    mean_exp_eigvals_unfilt[j][k] = mean_exp_eigvals_unfilt[j][k] + exp_eigvals_unfilt[i][j][k] / N_experiments

                for l in range(N_neibours_parameter):
                    mean_exp_mean_filt[j][k][l] = mean_exp_mean_filt[j][k][l] +exp_mean_filt[i][j][k][l]/N_experiments
                    mean_exp_std_filt[j][k][l] = mean_exp_std_filt[j][k][l] + exp_std_filt[i][j][k][l] / N_experiments
                    mean_exp_mean_unfilt[j][k][l] = mean_exp_mean_unfilt[j][k][l] + exp_mean_unfilt[i][j][k][l] / N_experiments
                    mean_exp_std_unfilt[j][k][l] = mean_exp_std_unfilt[j][k][l] + exp_std_unfilt[i][j][k][l] / N_experiments

    #Setting variables for max score
    max_score = 0.0
    max_exp = 0
    max_eps = 0
    max_neigh = 0
    max_filtOrUn = 0 # 0 for unfilt, 1 for filt
    #Calculate max scores
    for i in range(N_experiments):
        for j in range(N_dim_parameter):
            for k in range(N_epsilon_parameter):
                for l in range(N_neibours_parameter):
                    if (exp_mean_filt[i][j][k][l]>max_score):
                        max_score = exp_mean_filt[i][j][k][l]
                        max_exp = i
                        max_eps = k
                        max_neigh = l
                        max_filtOrUn = 1
                    if (exp_mean_unfilt[i][j][k][l]>max_score):
                        max_score = exp_mean_unfilt[i][j][k][l]
                        max_exp = i
                        max_eps = k
                        max_neigh = l
                        max_filtOrUn = 0
    # Setting variables for max score
    max_mean_score = 0.0

    max_mean_dim = 0
    max_mean_eps = 0
    max_mean_neigh = 0
    max_mean_filtOrUn = 0  # 0 for unfilt, 1 for filt
    # Calculate max scores
    for j in range(N_dim_parameter):
        for k in range(N_epsilon_parameter):
            for l in range(N_neibours_parameter):
                if (mean_exp_mean_filt[j][k][l] > max_mean_score):
                    max_mean_score = mean_exp_mean_filt[j][k][l]
                    max_mean_dim = j
                    max_mean_eps = k
                    max_mean_neigh = l
                    max_mean_filtOrUn = 1
                if (mean_exp_mean_unfilt[j][k][l] > max_mean_score):
                    max_mean_score = mean_exp_mean_unfilt[j][k][l]
                    max_mean_dim = j
                    max_mean_eps = k
                    max_mean_neigh = l
                    max_mean_filtOrUn = 0
    is_max_mean = 0
    #Display individual configurations
    dim_range = range(DIM_RNG_HIGH - DIM_RNG_LOW)
    for i in range(0,N_experiments):
        for k in range(N_epsilon_parameter):
            for l in range(N_neibours_parameter):
                plt.figure(i*N_epsilon_parameter*N_neibours_parameter+k*N_neibours_parameter+l,figsize=(16,9))
                config_string=' - Config: Eps='+str(epsilon_array[k])+' NNeighbors='+str(knn_neibours_array[l])
                file_config_string = '_Config_Eps_'+str(epsilon_array[k])+'_NNeighbors_'+str(knn_neibours_array[l])
                exp_string = 'Experiment '+ str(i)
                file_exp_string = 'Experiment_' + str(i)
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
                    SVM_patch = mpatches.Patch(color='red', label='SVM score')
                    DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                    plt.legend(handles=[DM_patch, SVM_patch])
                else:
                    DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                    plt.legend(handles=[DM_patch])
                title221='Prediction score as a function of dimension - Filtered'
                title223='Eigen values as a function of dimension - Filtered'
                if (max_exp==i and max_eps==k and max_neigh==l and max_filtOrUn==1):
                    title221 = title221 + ' - Max score'
                    title223 = title223 + ' - Max score'
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
                    SVM_patch = mpatches.Patch(color='red', label='SVM score')
                    DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                    plt.legend(handles=[DM_patch, SVM_patch])
                else:
                    DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                    plt.legend(handles=[DM_patch])

                title222 = 'Prediction score as a function of dimension - Unfiltered'
                title224 = 'Eigen values as a function of dimension - Unfiltered'
                if (max_exp==i and max_eps==k and max_neigh==l and max_filtOrUn==0):
                    title222 = title222 + ' - Max score'
                    title224 = title224 + ' - Max score'
                plt.title(title222)
                plt.subplot(224)
                axes = plt.gca()
                plt.scatter(range(1, DIM_RNG_HIGH), np.asarray(exp_eigvals_unfilt[i][-1][k]))
                for xy in zip(range(1,DIM_RNG_HIGH), np.asarray(exp_eigvals_unfilt[i][-1][k])):  # <--
                    axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
                plt.title(title224)
                #plt.show()
                if (SAVE_FIGS == 1):
                    plt.savefig(os.path.join(PATH_RESULTS_DIR, file_exp_string+file_config_string+'_MeanScoreAndStd.png'))
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
            x.append(mean_exp_mean_filt[0][k][l][0])
            for i in range(1,N_dim_parameter):
                x.append(mean_exp_mean_filt[i][k][l][0])
            y = []
            y.append(mean_exp_std_filt[0][k][l][0])
            for i in range(1,N_dim_parameter):
                y.append(mean_exp_std_filt[i][k][l][0])
            plt.errorbar(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x),
                         yerr=np.asarray(y), fmt='o', color='b', ecolor='navy')
            for xy in zip(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x)):  # <--
                axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
            plt.plot(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x), color='black')
            if svm == 1:
                plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH], [mean_svm_mean_filt, mean_svm_mean_filt], color='red')
                plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH],
                         [mean_svm_mean_filt + mean_svm_std_filt, mean_svm_mean_filt + mean_svm_std_filt],
                         color='firebrick')
                plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH],
                         [mean_svm_mean_filt - mean_svm_std_filt, mean_svm_mean_filt - mean_svm_std_filt],
                         color='firebrick')
                SVM_patch = mpatches.Patch(color='red', label='SVM score')
                DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                plt.legend(handles=[DM_patch, SVM_patch])
            else:
                DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                plt.legend(handles=[DM_patch])
            title221 = 'Prediction score as a function of dimension - Filtered '
            title223 = 'Eigen values as a function of dimension - Filtered'
            if ( max_mean_eps == k and max_mean_neigh == l and max_mean_filtOrUn == 1):
                title221 = title221 + ' - Max score'
                title223 = title223 + ' - Max score'
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
            x.append(mean_exp_mean_unfilt[0][k][l][0])
            for i in range(1,N_dim_parameter):
                x.append(mean_exp_mean_unfilt[i][k][l][0])
            y = []
            y.append(mean_exp_std_unfilt[0][k][l][0])
            for i in range(1,N_dim_parameter):
                y.append(mean_exp_std_unfilt[i][k][l][0])
            plt.errorbar(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x),
                         yerr=np.asarray(y), fmt='o', color='b', ecolor='navy')
            for xy in zip(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x)):  # <--
                axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
            plt.plot(range(DIM_RNG_LOW, DIM_RNG_HIGH), np.asarray(x), color='black')
            if svm == 1:
                plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH], [mean_svm_mean_unfilt, mean_svm_mean_unfilt], color='red')
                plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH],
                         [mean_svm_mean_unfilt + mean_svm_std_unfilt, mean_svm_mean_unfilt + mean_svm_std_unfilt],
                         color='firebrick')
                plt.plot([DIM_RNG_LOW, DIM_RNG_HIGH],
                         [mean_svm_mean_unfilt - mean_svm_std_unfilt, mean_svm_mean_unfilt - mean_svm_std_unfilt],
                         color='firebrick')
                SVM_patch = mpatches.Patch(color='red', label='SVM score')
                DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                plt.legend(handles=[DM_patch, SVM_patch])
            else:
                DM_patch = mpatches.Patch(color='blue', label='Diffusion map score')
                plt.legend(handles=[DM_patch])
            title222 = 'Prediction score as a function of dimension - Unfiltered '
            title224 = 'Eigen values as a function of dimension - Unfiltered'
            if (max_mean_eps == k and max_mean_neigh == l and max_mean_filtOrUn == 0):
                title222 = title222 + ' - Max score'
                title224 = title224 + ' - Max score'
            plt.title(title222)
            plt.subplot(224)
            axes = plt.gca()
            plt.scatter(range(1, DIM_RNG_HIGH), np.asarray(mean_exp_eigvals_unfilt[-1][k]))
            for xy in zip(range(1, DIM_RNG_HIGH), np.asarray(mean_exp_eigvals_unfilt[-1][k])):  # <--
                axes.annotate('(%.4f)' % xy[1], xy=xy, textcoords='data')
            plt.title(title224)
           # plt.show()
            if (SAVE_FIGS == 1):
                plt.savefig(os.path.join(PATH_RESULTS_DIR, file_exp_string + file_config_string + '_MeanScoreAndStd_Means.png'))
            plt.close()

def PrintLogToFiles(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt,svm=1):
    '''
    PrintLogToFiles:
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

    N_epsilon_parameter = int(math.log10(EPSILON_HIGH / EPSILON_LOW))  # requires running new DM
    epsilon_array = np.asarray([EPSILON_LOW * math.pow(10, i) for i in range(int(N_epsilon_parameter))])
    knn_neibours_array = np.asarray(range(N_NEIGHBORS_LOW, N_NEIGHBORS_HIGH, 2))
    N_neibours_parameter = len(knn_neibours_array)  # does not require neq
    N_experiments = len(full_paths)
    N_dim_parameter = DIM_RNG_HIGH - DIM_RNG_LOW

    # Write results to files
    f_seprate_analysis = open(os.path.join(PATH_RESULTS_DIR, SEPRATE_ANALYSIS_OUTPUT_FILE), 'w')
    for i in range(1):
        for k in range(N_epsilon_parameter):
            for l in range(N_neibours_parameter):
                f_seprate_analysis.write("Experiment - " + str(i + 1) + " - Config: Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Unfiltered\n")
                for j in range(N_dim_parameter):
                    f_seprate_analysis.write("Diffusion Dimension - " + str(j + 1) + "\n")
                    f_seprate_analysis.write("Accuracy mean=" + str(exp_mean_unfilt[i][j][k][l]) + "   STD=" + str(exp_std_unfilt[i][j][k][l]) + "\n")
                    f_seprate_analysis.write("Eigan Values - " + "\n")
                    f_seprate_analysis.write(str(exp_eigvals_unfilt[i][j][k]) + "\n")
                    f_seprate_analysis.write("-----------------------------" + "\n")
                if SVM == 1:
                    f_seprate_analysis.write("SVM baseline prediction rate- " + str(svm_mean_unfilt[i]) + "\n")
                f_seprate_analysis.write("==========================================================" + "\n")
                f_seprate_analysis.write("Experiment - " + str(i + 1) + " - Config: Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Filtered" + "\n")
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

    # Write results to files
    f_means_analysis = open(os.path.join(PATH_RESULTS_DIR, MEANS_ANALYSIS_OUTPUT_FILE), 'w')

    for k in range(N_epsilon_parameter):
        for l in range(N_neibours_parameter):
            f_means_analysis.write("Experiments means - Config: Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Unfiltered" + "\n")
            for j in range(N_dim_parameter):
                f_means_analysis.write("Diffusion Dimension - " + str(j + 1) + "\n")
                f_means_analysis.write("Accuracy mean=" + str(mean_exp_mean_unfilt[j][k][l][0]) + "   STD=" + str(mean_exp_std_unfilt[j][k][l][0]) + "\n")
                f_means_analysis.write("Eigan Values - " + "\n")
                f_means_analysis.write(str(mean_exp_eigvals_unfilt[j][k]) + "\n")
                f_means_analysis.write("-----------------------------" + "\n")
            if SVM == 1:
                f_means_analysis.write("SVM baseline prediction rate- " + str(mean_svm_mean_unfilt) + "\n")
            f_means_analysis.write("==========================================================" + "\n")
            f_means_analysis.write("Experiments means - Config: Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Filtered" + "\n")
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

def PrintLogToScreen(exp_mean_unfilt,exp_std_unfilt,exp_eigvals_unfilt,svm_mean_unfilt, svm_std_unfilt,exp_mean_filt,exp_std_filt,exp_eigvals_filt,svm_mean_filt, svm_std_filt,svm=1):
    '''
        PrintLogToScreen:
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
    N_epsilon_parameter = int(math.log10(EPSILON_HIGH / EPSILON_LOW))  # requires running new DM
    epsilon_array = np.asarray([EPSILON_LOW * math.pow(10, i) for i in range(int(N_epsilon_parameter))])
    knn_neibours_array = np.asarray(range(N_NEIGHBORS_LOW, N_NEIGHBORS_HIGH, 2))
    N_neibours_parameter = len(knn_neibours_array)  # does not require neq
    N_experiments = len(full_paths)
    N_dim_parameter = DIM_RNG_HIGH - DIM_RNG_LOW

    # First try printing results without writing to file
    for i in range(N_experiments):
        for k in range(N_epsilon_parameter):
            for l in range(N_neibours_parameter):
                print("Experiment - " + str(i + 1) + " - Config: Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Unfiltered")
                for j in range(N_dim_parameter):
                    print("Diffusion Dimension - " + str(j + 1))
                    print("Accuracy mean=" + str(exp_mean_unfilt[i][j][k][l]) + "   STD=" + str(exp_std_unfilt[i][j][k][l]))
                    print("Eigan Values - ")
                    print(str(exp_eigvals_unfilt[i][j][k]))
                    print("-----------------------------")
                if SVM == 1:
                    print("SVM baseline prediction rate- " + str(svm_mean_unfilt[i]))
                print("==========================================================")
                print("Experiment - " + str(i + 1) + " - Config: Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Filtered")
                for j in range(N_dim_parameter):
                    print("Diffusion Dimension - " + str(j + 1))
                    print("Accuracy mean=" + str(exp_mean_filt[i][j][k][l]) + "   STD=" + str(exp_std_filt[i][j][k][l]))
                    print("Eigan Values - ")
                    print(str(exp_eigvals_filt[i][j][k]))
                    print("-----------------------------")
                if SVM == 1:
                    print("SVM baseline prediction rate- " + str(svm_mean_filt[i]))
                print("==========================================================")


    # Calculate means and write results to screen
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
    # Try first to print before file printing
    for k in range(N_epsilon_parameter):
        for l in range(N_neibours_parameter):
            print("Experiments means - Config: Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Unfiltered")
            for j in range(N_dim_parameter):
                print("Diffusion Dimension - " + str(j + 1))
                print("Accuracy mean=" + str(mean_exp_mean_unfilt[j][k][l][0]) + "   STD=" + str(mean_exp_std_unfilt[j][k][l][0]))
                print("Eigan Values - ")
                print(str(mean_exp_eigvals_unfilt[j][k]))
                print("-----------------------------")
            if SVM == 1:
                print("SVM baseline prediction rate- " + str(mean_svm_mean_unfilt))
            print("==========================================================")
            print("Experiments means - Config: Eps=" + str(epsilon_array[k]) + " NNeighbours=" + str(knn_neibours_array[l]) + " - Filtered")
            for j in range(N_dim_parameter):
                print("Diffusion Dimension - " + str(j + 1))
                print("Accuracy mean=" + str(mean_exp_mean_filt[j][k][l][0]) + "   STD=" + str(mean_exp_std_filt[j][k][l][0]))
                print("Eigan Values - ")
                print(str(mean_exp_eigvals_filt[j][k]))
                print("-----------------------------")
            if SVM == 1:
                print("SVM baseline prediction rate- " + str(mean_svm_mean_filt))
            print("==========================================================")


################################### Main ######################################
if __name__ == '__main__':
    # --- Part A - loading saved data (working offline) --- #
    # ----------------------------------------------------- #
    file_names = os.listdir(RECORDS_PATH)
    full_paths = [os.path.join(RECORDS_PATH, fn) for fn in file_names if 'configuration' not in fn]

    if not os.path.exists(PATH_RESULTS_DIR):
        os.mkdir(PATH_RESULTS_DIR)

    # ======= Diffusion maps for data ========= #
    N_dim_parameter = DIM_RNG_HIGH - DIM_RNG_LOW # requires running new DM
    N_epsilon_parameter = int(math.log10(EPSILON_HIGH/EPSILON_LOW)) #requires running new DM
    epsilon_array = np.asarray([EPSILON_LOW*math.pow(10,i) for i in range(int(N_epsilon_parameter))])
    knn_neibours_array = np.asarray(range(N_NEIGHBORS_LOW,N_NEIGHBORS_HIGH,2))
    N_neibours_parameter = len(knn_neibours_array) # does not require neq
    N_experiments = len(full_paths)

    exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt = [[[[[] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)] , [[[[[] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)] , [[[[] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)]
    exp_mean_filt, exp_std_filt, exp_eigvals_filt = [[[[[] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)] , [[[[[] for i in range(N_neibours_parameter)] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)] , [[[[] for i in range(N_epsilon_parameter)] for i in range(N_dim_parameter)] for i in range(N_experiments)]
    svm_mean_unfilt = {}
    svm_std_unfilt = {}
    svm_mean_filt = {}
    svm_std_filt = {}

    variables_path = os.path.join(PATH_RESULTS_DIR, 'variables_' + str(EXPERIMENT_TYPE) + '.txt')

    if not os.path.exists(variables_path):
        for exp_ind, data_path in enumerate(full_paths):  # for each experiment
            eeg_fft_unfilt, eeg_fft_filt = load_and_filter_data(data_path)
            eeg_preprocessed_unfilt = preprocess_data(eeg_fft_unfilt, [5, 50]) # hardcoded frequency cut
            eeg_preprocessed_filt = preprocess_data(eeg_fft_filt, [5, 50]) # hardcoded frequency cut
            # =====Changes for a differnt experiment configuration ===== #
            labels, legend = get_aviv_exp_timeframes(eeg_preprocessed_unfilt)
            eeg_flatten_unfilt = np.asarray([x.flatten() for x in eeg_preprocessed_unfilt])
            eeg_flatten_filt = np.asarray([x.flatten() for x in eeg_preprocessed_filt])

            # ==== Using only Open / Close data ==== #
            if SKIP_BLINK == 1:
                data_0_unfilt, labels_0 = extract_state(eeg_flatten_unfilt, labels, 0)
                data_2_unfilt, labels_2 = extract_state(eeg_flatten_unfilt, labels, 2)
                data_0_filt, labels_0 = extract_state(eeg_flatten_filt, labels, 0)
                data_2_filt, labels_2 = extract_state(eeg_flatten_filt, labels, 2)
                data_02_filt = data_0_filt + data_2_filt
                data_02_unfilt = data_0_unfilt + data_2_unfilt
                labels_02 = labels_0 + labels_2
            # other embedding visualization methods
            if SKIP_BLINK == 1:
                # visualize_eeg(data_02_unfilt[:-1], labels_02[:-1], legend, FS, domain='Freq', meth='all', dim=2, exp_ind=exp_ind, filtOrUn=0)
                # visualize_eeg(data_02_filt[:-1], labels_02[:-1], legend, FS, domain='Freq', meth='all', dim=2, exp_ind=exp_ind, filtOrUn=1)
                do_nothing = 5
            else:
                visualize_eeg(eeg_flatten_unfilt[:-1], labels[:-1], legend, FS, domain='Freq', meth='all', dim=2,exp_ind=exp_ind, filtOrUn=0)
                visualize_eeg(eeg_flatten_filt[:-1], labels[:-1], legend, FS, domain='Freq', meth='all', dim=2, exp_ind=exp_ind, filtOrUn=1)
            # ====================================== #
            #cur_mean_unfilt, cur_std_unfilt, cur_eig_unfilt = [], [], []
            #cur_mean_filt, cur_std_filt, cur_eig_filt = [], [], []

            for epsilon_index in range(N_epsilon_parameter):
                #cur_mean_unfilt, cur_std_unfilt, cur_eig_unfilt = [], [], []
                #cur_mean_filt, cur_std_filt, cur_eig_filt = [], [], []
                epsilon = epsilon_array[epsilon_index]  # diffusion distance epsilon
                dim_index=0
                for dimension in range(DIM_RNG_LOW, DIM_RNG_HIGH):
                    coords_out_unfilt, labels_out_unfilt = [], []
                    coords_out_filt, labels_out_filt = [], []
                    if SKIP_BLINK == 1:
                        coords_unfilt, dataList, eigvals_unfilt = dm.diffusionMapping(data_02_unfilt[:-1],
                                                           lambda x, y: math.exp(-LA.norm(x - y) / epsilon),
                                                           t=2, dim=dimension)
                        labels_out_unfilt.append(labels_02[:-1])
                        coords_filt, dataList, eigvals_filt = dm.diffusionMapping(data_02_filt[:-1],
                                                                        lambda x, y: math.exp(-LA.norm(x - y) / epsilon),
                                                                        t=2, dim=dimension)
                        labels_out_filt.append(labels_02[:-1])
                        # show_diffusion(coords_unfilt, labels_out_unfilt, legend)
                        # show_diffusion(coords_filt, labels_out_filt, legend)
                    else:
                        coords_unfilt, dataList, eigvals_unfilt = dm.diffusionMapping(eeg_flatten_unfilt[:-1],
                                                                        lambda x, y: math.exp(-LA.norm(x - y) / epsilon),
                                                                        t=2, dim=dimension)
                        labels_out_unfilt.append(labels[:-1])
                        coords_filt, dataList, eigvals_filt = dm.diffusionMapping(eeg_flatten_filt[:-1],
                                                                        lambda x, y: math.exp(-LA.norm(x - y) / epsilon),
                                                                        t=2, dim=dimension)
                        labels_out_filt.append(labels[:-1])
                        # show_diffusion(coords_unfilt, labels_out_unfilt, legend)
                        # show_diffusion(coords_filt, labels_out_filt, legend)
                    #coords_out_unfilt.append(coords_unfilt)
                    #pred_mean_unfilt, pred_std_unfilt = knn_clustering(coords_out_unfilt, labels_out_unfilt)
                    #cur_mean_unfilt.append(pred_mean_unfilt)
                    #cur_std_unfilt.append(pred_std_unfilt)
                    #cur_eig_unfilt.append(eigvals_unfilt)

                    #coords_out_filt.append(coords_filt)
                    #pred_mean_filt, pred_std_filt = knn_clustering(coords_out_filt, labels_out_filt)
                    #cur_mean_filt.append(pred_mean_filt)
                    #cur_std_filt.append(pred_std_filt)
                    #cur_eig_filt.append(eigvals_filt)

                    coords_out_unfilt.append(coords_unfilt)
                    coords_out_filt.append(coords_filt)
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
            #exp_mean_unfilt[exp_ind] = cur_mean_unfilt
            #exp_std_unfilt[exp_ind] = cur_std_unfilt
            #exp_eigvals_unfilt[exp_ind] = cur_eig_unfilt
            #exp_mean_filt[exp_ind] = cur_mean_filt
            #exp_std_filt[exp_ind] = cur_std_filt
            #exp_eigvals_filt[exp_ind] = cur_eig_filt
            if SVM == 1:    ## for each experiment once
                svm_pred_mean_unfilt, svm_pred_std_unfilt = svm_cross_val(data_02_unfilt, labels_02)
                svm_mean_unfilt[exp_ind] = svm_pred_mean_unfilt
                svm_std_unfilt[exp_ind] = svm_pred_std_unfilt
                svm_pred_mean_filt, svm_pred_std_filt = svm_cross_val(data_02_filt, labels_02)
                svm_mean_filt[exp_ind] = svm_pred_mean_filt
                svm_std_filt[exp_ind] = svm_pred_std_filt
            print('Finished calculating diffusion maps and SVM for expeiment - ' + str(exp_ind))
        # ===== Report results and print them to the screen ===== #
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
                        'eigvals_unf': exp_eigvals_unfilt
                        }
        pickle.dump(results_dict, file)
        file.close()
    else:
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
        file.close()
    # write_xml(os.path.join(PATH_RESULTS_DIR, 'results.xlsx'), exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt,
    #           N_experiments, N_epsilon_parameter, N_neibours_parameter, N_dim_parameter,
    #           SVM=True, mean_svm=svm_mean_unfilt, filtered=False)
    # write_xml(os.path.join(PATH_RESULTS_DIR, 'results.xlsx'), exp_mean_filt, exp_std_filt, exp_eigvals_filt,
    #           N_experiments, N_epsilon_parameter, N_neibours_parameter, N_dim_parameter,
    #           SVM=True, mean_svm=svm_mean_filt, filtered=True)
    # ======================================================== #

    plotErrorMeanAndStdPerExpAll(exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt, svm_mean_unfilt, svm_std_unfilt,exp_mean_filt, exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt, exp_ind+1, SVM)
    #plotErrorMeanAndStdPerExp(exp_mean_filt, exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt, exp_ind+1, SVM)

    PrintLogToFiles(exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt, svm_mean_unfilt, svm_std_unfilt, exp_mean_filt, exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt, svm=1)
    PrintLogToScreen(exp_mean_unfilt, exp_std_unfilt, exp_eigvals_unfilt, svm_mean_unfilt, svm_std_unfilt, exp_mean_filt, exp_std_filt, exp_eigvals_filt, svm_mean_filt, svm_std_filt, svm=1)
    # ===== Calculating clusters centers ===== #
    # pred_mean, pred_std = knn_clustering(coords_out, labels_out)

    ttt= 5
# run a blank test

# write to excel the results of the experiments





