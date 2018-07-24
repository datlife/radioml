import commpy as cp
import numpy as np
from tensorflow.keras.models import load_model
from commpy.modulation import Modem


class Receiver(object):
    """Abstract Class of a Radio Receiver, which takes in noisy inputs to
    estimate original message bits. 
    
    It might be an end-to-end system or a combination of of a Equalizier, 
    Demodulator, and Decoder.
    """
    def __init__(self):
        pass

    def __call__(self, noisy_signals):
        """Process noisy signals

        Arguments:
            noisy_signals: 

        Return:
            decoded_inputs: estimate of original message bits
        """
        raise NotImplementedError


class ModularReceiver(Receiver):
    """Modular Receiver represents a radio model that consists of 3 parts:
        * Equalizer (Not implemented yet)
        * Demodulator (QPSK, QAM16, PSK, or Neural-network based model)
        * Decoder (Viterbi, MAP, or Neural-network based model)
    #@TODO: add Equalizer

    # Arguments:
        demodulator: `Demodulator` object
        decoder: `
    """
    def __init__(self, demodulator, decoder):
        super(ModularReceiver, self).__init__()
        self.demodulator = demodulator
        self.decoder = decoder

    def __call__(self, noisy_inputs, in_batch=False):
        if in_batch:
            batch_size = len(noisy_inputs)
            coded_bits  = self.demodulator.demodulate(noisy_inputs.ravel())
            coded_bits = coded_bits.reshape((batch_size, -1))
        else:
            coded_bits  = self.demodulator.demodulate(noisy_inputs)

        decoded_bits = self.decoder.decode(coded_bits)

        return decoded_bits


class End2EndReceiver(Receiver):
    """End-to-End Receiver."""
    def __init__(self, model, data_length):
        super(End2EndReceiver, self).__init__()
        self.data_length = data_length
        self.model = load_model(model, compile=False)

    def __call__(self, noisy_inputs, batch_size):
        preprocessed_inputs = self._preprocess_fn(noisy_inputs)
        predictions = self.model.predict(preprocessed_inputs, batch_size)
        return np.squeeze(predictions, -1).round()


    def _preprocess_fn(self, complex_inputs):
        """Encode complex inputs to 2D ndarray inputs"""
        x = np.stack((np.array(complex_inputs).real,
                      np.array(complex_inputs).imag),
                      axis=-1)
        return x.reshape((-1, data_length, 2))    
