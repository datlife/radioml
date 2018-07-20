import commpy as cp
import numpy as np
from tensorflow.keras.models import Model
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
    """Classic Receiver, consists of a Demodulator, and Viterbi
    Decoder

    #@TODO: add Equalizer
    """
    def __init__(self, demodulator, decoder):
        super(ModularReceiver, self).__init__()
        self.demodulator = demodulator
        self.decoder = decoder

    def __call__(self, noisy_inputs):
        coded_bits  = self.demodulator.demodulate(noisy_inputs)
        decoded_bits = self.decoder.decode(coded_bits)

        return decoded_bits


class End2EndReceiver(Receiver):
    """End-to-End Receiver."""
    def __init__(self, model):
        super(End2EndReceiver, self).__init__()
        self.model = model

    def __call__(self, noisy_inputs):
        x = self.model.predict(noisy_inputs)
        return x
