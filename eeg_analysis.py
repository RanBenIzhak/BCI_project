from scipy import signal, fftpack
import numpy as np
from numpy import linalg as LA
import os
import matplotlib.pyplot as plt
import diffusion_maps as dm
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import cross_val_score
from sklearn import manifold, decomposition, svm


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
# ===== KNN constants ==== #
N_NEIGHBORS = 3
n_components = 5 #ALL
n_neighbors = 5  #LLE, Isomap
# ==== Diffusion parameters ==== #
DIM_RNG_LOW = 1
DIM_RNG_HIGH = 6

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

# def visualize_eeg(eeg_data, labels, legend, fs, domain='Freq', meth='tsne', dim=2):
#     '''
#     Visualize the input eeg data in time domain
#     :param eeg_data: 2d time domain eeg data - [samples, channels]
#     :param fs: sample frequency (for axis labels)
#     :param labels: labels of eeg dataset according to labels
#     :param legend: legend of data and labels tagging
#     :param domain: 'Time'/ 'Freq' - the domain we want to visualize in. Note: Data is recived in frequency domain and entering time does not take mean into account
#     :param meth: 'tsne'/ 'lle'/ 'pca'/ 'isomap' - the method used for data embedding in lower dimensional space
#     :param dim: 2(default)/ 3 - the dimenssion to show the data in
#     :return: None
#     '''
#
#     # preform lower dimenstional embbeding and show the results
#     if meth=='tsne':
#  #       print eeg_data.shape
#  #       eeg_data_float64 = np.asarray(eeg_data).astype('float64')
#  #       eeg_data_float64 = eeg_data_float64.reshape((eeg_data_float64.shape[0], -1))
#         X_embedded=manifold.TSNE(dim).fit_transform(eeg_data)
#     elif meth=='lle':
#         X_embedded= manifold.LocallyLinearEmbedding(n_neighbors, dim,eigen_solver='auto', method='standard').fit_transform(X)
#     elif meth=='pca':
#         X_embedded=decomposition.PCA(dim).fit_transform(eeg_data)
#     elif meth=='isomap':
#         X_embedded=manifold.Isomap(n_neighbors, dim).fit_transform(eeg_data)
#     # If we want to visualize in frequency domain/time domain
#     # if domain=='Time':
#     #     X_embedded=np.fft.ifftshift(fftpack.ifft((eeg_data[i * start_diff:i * start_diff + SAMPLES_PER_FRAME, :]),
#     #                         axes=1))
#     #Plot the data visualization
#     # fig = plt.figure()
#     if dim==2:
#         show_embedded(X_embedded,labels,legend)
#     elif dim==3:
#         fig = plt.figure()
#         show_3D(X_embedded,labels,fig,meth,domain)
#     else:
#         print("Please set dimension to 2/3 for visualization")

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
    samples_per_mode = int(20 / (TIME_FRAME * (1 - OVERLAP)))
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

def knn_clustering(data, labels, neighbors_num=N_NEIGHBORS):
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

def show_embedded_all(coordinates_tsne,coordinates_lle,coordinates_pca, coordinates_isomap, labels, legend):
    colors = ['red', 'green', 'blue']
    fig = plt.figure()
    ax_221 = fig.add_subplot(221)
    ax_221.set_title('Tsne')
    a = np.asarray(coordinates_tsne)
    x = a[:, 0]
    y = a[:, 1]
    for label in legend:
        cur_label = legend[label]
        plt.scatter(x[labels==label], y[labels==label], c=colors[label], label=cur_label)
    ax_222 = fig.add_subplot(222)
    ax_222.set_title('LLE')
    a = np.asarray(coordinates_lle)
    x = a[:, 0]
    y = a[:, 1]
    for label in legend:
        cur_label = legend[label]
        plt.scatter(x[labels == label], y[labels == label], c=colors[label], label=cur_label)
    ax_223 = fig.add_subplot(223)
    ax_223.set_title('PCA')
    a = np.asarray(coordinates_pca)
    x = a[:, 0]
    y = a[:, 1]
    for label in legend:
        cur_label = legend[label]
        plt.scatter(x[labels == label], y[labels == label], c=colors[label], label=cur_label)
    ax_224 = fig.add_subplot(224)
    ax_224.set_title('Isomap')
    a = np.asarray(coordinates_isomap)
    x = a[:, 0]
    y = a[:, 1]
    for label in legend:
        cur_label = legend[label]
        plt.scatter(x[labels == label], y[labels == label], c=colors[label], label=cur_label)
    plt.legend(loc=2)
    plt.suptitle('2-D Visualization of eeg data')
    plt.show()

def visualize_eeg(eeg_data, labels, legend, fs, domain='Freq', meth='tsne', dim=2):
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
        show_embedded_all(X_embedded_tsne, X_embedded_lle, X_embedded_pca, X_embedded_isomap, labels, legend)

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

if __name__ == '__main__':
    # --- Part A - loading saved data (working offline) --- #
    # ----------------------------------------------------- #
    cur_path = os.path.dirname(os.path.realpath(__file__))
    records_path = os.path.join(cur_path, 'Data')
    # records_path = 'C:\\Users\\ranbenizhak\\OneDrive\\BCI\\BCI_project\\Data\\'

    file_names = os.listdir(records_path)
    full_paths = [os.path.join(records_path, fn) for fn in file_names if 'Aviv' in fn]

    # ======= Diffusion maps for data ========= #
    exp_mean, exp_std, exp_eigvals = {}, {}, {}
    svm_mean = {}
    for exp_ind, data_path in enumerate(full_paths):  # for each experiment
        eeg_fft_unfilt, eeg_fft_filt = load_and_filter_data(data_path)
        eeg_preprocessed = preprocess_data(eeg_fft_unfilt, [5, 50])
        labels, legend = get_aviv_exp_timeframes(eeg_preprocessed)
        eeg_flatten = np.asarray([x.flatten() for x in eeg_preprocessed])

        # ==== Using only Open / Close data ==== #
        data_0, labels_0 = extract_state(eeg_flatten, labels, 0)
        data_2, labels_2 = extract_state(eeg_flatten, labels, 2)
        data_02 = data_0 + data_2
        labels_02 = labels_0 + labels_2
        # ====================================== #
        cur_mean, cur_std, cur_eig = [], [], []
        for dimension in range(DIM_RNG_LOW, DIM_RNG_HIGH):
            coords_out, labels_out = [], []
            # other embedding visualization methods
            # visualize_eeg(eeg_flatten[:-1], labels[:-1], legend, FS, domain='Freq', meth='tsne', dim=3)
            epsilon = 1000  # diffusion distance epsilon
            coords, dataList, eigvals = dm.diffusionMapping(data_02[:-1],
                                                   lambda x, y: math.exp(-LA.norm(x - y) / epsilon),
                                                   t=2, dim=dimension)
            labels_out.append(labels_02[:-1])
            coords_out.append(coords)
            pred_mean, pred_std = knn_clustering(coords_out, labels_out)
            cur_mean.append(pred_mean)
            cur_std.append(pred_std)
            cur_eig.append(eigvals)

        exp_mean[exp_ind] = cur_mean
        exp_std[exp_ind] = cur_std
        exp_eigvals[exp_ind] = cur_eig

        svm_pred_mean, svm_pred_std = svm_cross_val(data_02, labels_02)
        svm_mean[exp_ind] = svm_pred_mean
        # show_diffusion(coords_out[0], labels_out[0], legend)

    for j in range(4):
        print("Experiment - " + str(j+1))
        for i in range(DIM_RNG_HIGH - DIM_RNG_LOW):
            print("Diffusion Dimension - " + str(i+1))
            print("Accuracy mean=" + str(exp_mean[j][i]) + "   STD=" + str(exp_std[j][i]))
            print("Eigan Values - ")
            print(str(exp_eigvals[j][i]))
            print("-----------------------------")
        print("SVM baseline prediction rate- " + str(svm_mean[j]))
        print("==========================================================")



    # ===== Calculating clusters centers ===== #
    # pred_mean, pred_std = knn_clustering(coords_out, labels_out)
    ttt = 5

# run a blank test

# write to excel the results of the experiments






