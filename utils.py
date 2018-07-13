import numpy as np
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
def get_scores(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    hscore = hamming_score(y_pred, y_true)
    return accuracy, hscore

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

# Find average hamming distance. Inputs should be np array of integers.
def hamming_score(y_hat, y_true):
    assert y_hat.shape == y_true.shape
    assert len(y_hat.shape) == 1 == len(y_true.shape)

    n = y_hat.shape[0]
    total_dist = 0
    for i in range(n):
        total_dist += hamming_dist(y_hat[i], y_true[i])
    return round(total_dist / n, 6)

#### VISUALIZATION FUNCTIONS ####
def visualize_nn_output(inputs, predicted_labels, M):
    x = inputs[:,0] # first column/real part
    y = inputs[:,1] # second column/imaginary part
    plt.scatter(x, y, c=predicted_labels, cmap='Set1')
    plt.colorbar()
    plt.show()


def visualize(signals, constellation, ax, predictions=None, cmap=pylab.cm.Spectral):
    ax.scatter(np.real(signals), np.imag(signals), 
               c=predictions if predictions is not None else None, 
               cmap=cmap if predictions is not None else None)
    ax.scatter(np.real(constellation), np.imag(constellation),color='yellow')
    ax.axhline(0)
    ax.axvline(0)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


