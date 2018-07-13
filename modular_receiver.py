import numpy as np
import tensorflow  as tf
from tensorflow.keras.layers import Dense, Input

class ModularReceiver(object):
    """Simulate a two parts: Demod and Decoder."""

    def __init__(self, demod_model_path, decoder_model_path):
        self.demod_model = tf.keras.models.load_model(demod_model_path)
        self.rnn_model = tf.keras.models.load_model(decoder_model_path, 
            custom_objects={'BER': BER, 'BLER': BLER})

    def demodulate(self, complex_inputs):

        inputs = self._preprocess_demod_inputs(complex_inputs)
        return np.argmax(self.demod_model.predict(inputs), -1)

    def decode(self, inputs):
        return np.argmax(self.rnn_model.predict(inputs), -1)

    def _preprocess_demod_inputs(self, complex_inputs):
        """Encode complex into a 2D array inputs"""
        x = np.stack((np.array(complex_inputs).real,
                     np.array(complex_inputs).imag),
                      axis=-1)
        preprocessed_inputs = x.reshape((-1, 2))
        return preprocessed_inputs

def BER(y, y_pred):
  """Measure Bit Error Rate (BER)
  Args:
    y - tf.Tensor - shape (batch_size, K, 1)
    y_pred - tf.Tensor - shape (batch_size, K, 1)

  Returns:
    ber - a tf.float - represents bit error rate
        in a batch.
  """
  hamming_distances =  tf.cast(tf.not_equal(y, tf.round(y_pred)), tf.int32)
  ber = tf.reduce_sum(hamming_distances) / tf.size(y)
  return ber

def BLER(y, y_pred):
    """Measure Bit Block Error Rate (BER)
    Args:
      y - tf.Tensor - shape (batch_size, K, 1)
      y_pred - tf.Tensor - shape (batch_size, K, 1)

    Returns:
      bler - a tf.float - represents bit block error rate
          in a batch.
    """
    num_blocks_per_batch = tf.cast(tf.shape(y)[0], tf.int64)
    hamming_distances =  tf.cast(tf.not_equal(y, tf.round(y_pred)), tf.int32)
    bler = tf.count_nonzero(tf.reduce_sum(hamming_distances, axis=1)) / num_blocks_per_batch
    return bler
