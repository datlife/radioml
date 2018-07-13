"""Simulate Convolutional Encoded Signals over AWGN Channel."""
import numpy as np 
import commpy as cp
import multiprocessing as mp 
from commpy.channelcoding import Trellis

class SignalGenerator(object):
    """Simulate how message bits are encoded, modulated and 
    sent over AWGN Channel.

    Args:
        modem: commpy.modulation.Modem object 
            can be a Phase Shift Keying (PSK) or 
            Quadrature Amplitude Modulation (QAM) Modem

    Examples:
        >> import commpy as cp
        >> qam64_modem = cp.modulation.QAMModem(m=64)
        >> Simulator = SignalGenerator(modem=quam64_modem)

        >> message_bits, clean_signals, noisy_signals = Simulator(
                num_sequences=100, 
                bit_stream_lengh=50,
                snr_in_dB=10.0)

    Assumptions:
        * Message bits are encoded in Convolutional Coding Scheme
    """

    # A Trellis object to encode message bits
    trellis = Trellis(memory   = np.array([2]), 
                      g_matrix = np.array([[0o7, 0o5]]), 
                      feedback = 0o7, 
                      code_type= 'rsc')

    def __init__(self, modem):
        self.modem = modem

    def __call__(self, num_sequences, bit_stream_length, snr_in_dB):
        """Generate a sequence of noisy complex numbers

        Args:
            num_sequences - int - number of sequences
            bit_stream_length - int - length of one message bit
            snr_in_dB - float - 
                represent signal to noise ratio in Decibels

        Returns:
            A tuple of \ 
            (original message bits, modulated clean signals, modulated noisy singals)
        """
        orignal_msg_bits, moded_bits, noisy_outputs = [], [], []

        with mp.Pool(mp.cpu_count()) as pool:
            result = pool.starmap(simulation_func,
                iterable=[(bit_stream_length, self.modem, self.trellis, snr_in_dB) \
                          for i in range(num_sequences)])
            orignal_msg_bits, moded_bits, noisy_outputs = zip(*result)
        return (np.array(orignal_msg_bits), np.array(moded_bits),  np.array(noisy_outputs))


def simulation_func(bit_stream_length, modem, trellis, snr):
    message_bits = generate_message_bits(bit_stream_length)
    conv = cp.channelcoding.convcode.conv_encode(message_bits, trellis)
    moded = modem.modulate(conv)
    bits = cp.channels.awgn(moded, snr_dB=snr)
    return message_bits, moded, bits

def generate_message_bits(bit_stream_length, p=0.5):
    """Generate message bits length `seq_len` of a random binary 
    sequence, where each bit picked is a one with probability p.
    Args:
        bit_stream_length: - int - length of a message bit
        p - float - probability
    Return:
        seq: - 1D ndarray - represent a message bits
    """
    seq = np.zeros(bit_stream_length)
    for i in range(bit_stream_length):
        seq[i] = 1 if (np.random.random() < p) else 0
    return seq  

