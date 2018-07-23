import pylab
import numpy as np
import commpy as cp
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from sklearn.metrics import accuracy_score

# snr is in decibels
def add_noise(symbol, num_copies = 1, snr = 0.):
    #Add iid gaussian noise with standard deviation sigma = 10**(snr/10)
    noise_std = 10**(-snr/20.0)
    noise_vec = noise_std*(np.random.randn(num_copies,)+ 1j*np.random.randn(num_copies,))
    symbol_noisy = symbol + noise_vec
    return symbol_noisy

#### UTIL FUNCTIONS ####
def get_ber_bler(estimated_bits, original_bits):
    n_sequences = len(original_bits)
    hamming_distances = []
    for i in range(n_sequences):       
        dist = cp.utilities.hamming_dist(
            original_bits[i].astype(int),
            estimated_bits[i].astype(int))
        hamming_distances.append(dist)
    ber = np.sum(hamming_distances) / np.product(np.shape(original_bits))
    bler = np.count_nonzero(hamming_distances) / len(original_bits)
    return ber, bler
    
def get_scores(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    hscore = hamming_score(y_pred, y_true)
    return accuracy, hscore

# Find average hamming distance. Inputs should be np array of integers.
def hamming_score(y_hat, y_true):
    assert y_hat.shape == y_true.shape
    assert len(y_hat.shape) == 1 == len(y_true.shape)

    n = y_hat.shape[0]
    total_dist = 0
    for i in range(n):
        total_dist += hamming_dist(y_hat[i], y_true[i])
    return round(total_dist / n, 6)


def hamming_dist(x1, x2):
    '''
    Return hamming distance between two integers. 
    Extends both to equal bit size if not already the case.
    '''
    x1 = [_ for _ in bin(x1)[2:]]
    x2 = [_ for _ in bin(x2)[2:]]
    x_short = min((x1, x2), key=lambda x: len(x))
    x_long = max((x1, x2), key=lambda x: len(x))
    diff = len(x_long) - len(x_short)
    x_short = ['0']*diff + x_short
    assert len(x_short) == len(x_long)
    return hamming(x_short, x_long) * len(x_short)


#### VISUALIZATION FUNCTIONS ####
def visualize_nn_output(inputs, predicted_labels, M):
    x = inputs[:,0] # first column/real part
    y = inputs[:,1] # second column/imaginary part
    plt.scatter(x, y, c=predicted_labels, cmap='Set1')
    plt.colorbar()
    plt.show()


def visualize_demodulation(signals, constellation, ax, title='', predictions=None, cmap=pylab.cm.Spectral):
    ax.scatter(np.real(signals), np.imag(signals), 
               c=predictions if predictions is not None else None, 
               cmap=cmap if predictions is not None else None)
    ax.scatter(np.real(constellation), np.imag(constellation),color='yellow')
    ax.axhline(0)
    ax.axvline(0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def visualize_acc_err(ax1, ax2, errors_logs, accuracies_logs, snr_range):
    ax1.plot(snr_range, np.array(errors_logs).T[1, :], 'r-')
    ax1.plot(snr_range, np.array(errors_logs).T[0, :], 'b--*')
    ax1.legend(['NN Demod','Baseline'], fontsize=15)
    ax1.set_title('Error')
    ax1.set_xlabel('SNR (in dB)')
    ax1.grid(True,'both')
    ax1.set_xlim(np.min(snr_range), np.max(snr_range))
    ax2.plot(snr_range, np.array(accuracies_logs).T[1,:], 'r-')
    ax2.plot(snr_range, np.array(accuracies_logs).T[0,:], 'b--*')
    ax2.legend(['NN Demod', 'Baseline'], fontsize=15)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('SNR (in dB)')
    ax2.set_xlim(np.min(snr_range), np.max(snr_range))

def visualize_ber_bler(ax1, ax2, ber_logs, bler_logs, snr_range):
    ax1.plot(snr_range, np.array(ber_logs).T[1, :], 'r-')
    ax1.plot(snr_range, np.array(ber_logs).T[0, :], 'b--*')
    ax1.legend(['NN Decoder','Baseline'], fontsize=15)
    ax1.set_xlabel('SNR (in dB)')
    ax1.set_ylabel('Bit Error Rate (BER)')
    ax1.grid(True,'both')
    ax1.set_xlim(np.min(snr_range), np.max(snr_range))
    ax2.plot(snr_range, np.array(bler_logs).T[1,:], 'r-')
    ax2.plot(snr_range, np.array(bler_logs).T[0,:], 'b--*')
    ax2.legend(['NN Decoder', 'Baseline'], fontsize=15)
    ax2.set_xlabel('SNR (in dB)')
    ax2.set_ylabel('Block Error Rate (BLER)')
    ax2.set_xlim(np.min(snr_range), np.max(snr_range))
