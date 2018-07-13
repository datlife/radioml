import numpy as np
import tensorflow  as tf
from tensorflow.keras.layers import Dense, Input

class ModularReceiver(object):
    """Simulate a two parts: Demod and Decoder."""

    def __init__(self, num_classes, demod_model_path):
        self.demod_model = tf.keras.models.load_model(demod_model_path)
        # self.rnn_model = tf.keras.models.load_model(decoder_model_path)

    def demodulate(self, complex_inputs):

        inputs = self._preprocess_demod_inputs(complex_inputs)
        return np.argmax(self.demod_model.predict(inputs), -1)

    def decode(self, inputs):
        pass

    def _preprocess_demod_inputs(self, complex_inputs):
        """Encode complex into a 2D array inputs"""
        x = np.stack((np.array(complex_inputs).real,
                     np.array(complex_inputs).imag),
                      axis=-1)
        preprocessed_inputs = x.reshape((-1, 2))
        return preprocessed_inputs
