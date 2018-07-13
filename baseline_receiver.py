"""Contains Implementation of Optical Demodulator. 

Used for benchmarking against Modular Neural Demod and 
End2End Neural Receiver (Demod + ECC).

NOTES: Some codes are based on previous work in:
https://github.com/vignesh-subramanian/ml-demodulation

"""
import numpy as np
import commpy as cp
import multiprocessing as mp


class BaselineReceiver(object):
    """Baseline Radio Receiver with classical Demod and Viterbi 
    Decoder.
    """

    def __init__(self, modem, trellis):
        """Intialize Radio Receiver

        Args:
            modem: commpy.modulation.Modem object 
                can be a Phase Shift Keying (PSK) or 
                Quadrature Amplitude Modulation (QAM) Modem

            trellis: a Trellis object
        """
        self.modem = modem
        self.trellis =trellis

    def demodulate(self, complex_inputs):
        """Classify complex inputs into its symbol in constellation

        Args:
            input: - a complex 1D array- 
                represent message bits in complex number
        Return:
            predictions:
        """
        constellation = self.modem.constellation
        predictions = [np.argmin(np.abs(i - constellation)) for i in complex_inputs]
        return np.array(predictions)

    def decode(self, inputs):
        """Decode input singals

        Args:
            inputs: 1D array in float

        Return:
            decoded_inputs: estimate of original message bits
        """
        return viterbi_decoder(inputs, self.trellis)



def viterbi_decoder(noisy_bits, trellis):
    """Decode Convolutional Codes with Viterbi Algorithm.

    Args:
        noisy_bits : demodulated bits
    """
    decoded_bits = cp.channelcoding.viterbi_decode(
        coded_bits=noisy_bits.astype(float), 
        trellis=trellis,
        tb_depth=15,
        decoding_type='hard')
    
    return decoded_bits
