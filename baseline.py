"""Contains Implementation of Optical Demodulator. 

Used for benchmarking against Modular Neural Demod and 
End2End Neural Receiver (Demod + ECC).

NOTES: Some codes are based on previous work in:
https://github.com/vignesh-subramanian/ml-demodulation

"""
import numpy as np
import utils as u


def optimal_demodulator(inputs, constellation):
    """Classify input dataset X

    Args:
        input: - a complex 1D array- 
            represent message bits in complex number
        constellation : a complex 1D array
            represent constellation in a Modem (QAM16, QAM64, etc)

    Return:
        predictions:
    """
    predictions = [np.argmin(np.abs(point - constellation)) for point in inputs]
    return np.array(predictions)


def optimal_probabilistic_demodulator(x, symbol_function, M):
    '''Expect x to be a BATCH of x-valued numbers.'''
    assert x.ndim == 2, 'x should be batch but x.ndim = {}'.format(x.ndim)
    symbols = np.array([symbol_function(i, M) for i in range(M)])
    x_complex = np.array([w[0] + 1j*w[1] for w in x])
    distances = np.zeros(shape=(len(x), M))
    for i in range(len(x)): 
        for j in range(M): 
            distances[i][j] = -1 * np.linalg.norm(x_complex[i] - symbols[j])
    exp_neg_distances = np.exp(distances)
    row_sums = np.repeat(np.expand_dims(np.sum(exp_neg_distances, axis = 1), -1), 4, axis=1)
    probs = np.divide(exp_neg_distances, row_sums)
    return probs