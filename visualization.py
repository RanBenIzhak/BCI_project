import matplotlib.pyplot as plt

def visualize_eeg(eeg_data, fs, domain='Time'):
    '''
    Visualize the input eeg data in time domain
    :param eeg_data: 2d time domain eeg data - [samples, channels]
    :param fs: sample frequency (for axis labels)
    :param domain: 'Time'/ 'Freq' - the domain to show the data in
    :return: None
    '''
    fig = plt.figure()


