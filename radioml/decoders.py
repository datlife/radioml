"""Contains implementations of Viterbi Decoder Algorithm and 
Neural Decoder [1] with bi-directional GRU architecture.

[1] paper.

"""
import numpy as np
from tensorflow.keras.models import load_model
from commpy.channelcoding import viterbi_decode

class Decoder(object):
    """Abstract Class for a Decoder Object."""
    def __init__(self):
        pass
    def decode(self, inputs):
        pass


class ViterbiDecoder(Decoder):
    def __init__(self, trellis, tb_depth=15, decoding_type='hard'):
        super(ViterbiDecoder, self).__init__()
        self.trellis = trellis
        self.tb_depth = tb_depth
        self.decoding_type= decoding_type
    def decode(self, inputs):
        return viterbi_decode(inputs, self.trellis, self.tb_depth, 
                              self.decoding_type)


class NeuralDecoder(Decoder):
    def __init__(self, model_path, block_length):
        super(NeuralDecoder, self).__init__()
        self.model = load_model(model_path, compile=False)
        self.block_length = block_length

    def decode(self, inputs):
        predictions =  self.model.predict(self._preprocess_fn(inputs))
        return  np.squeeze(predictions, -1).round()
    
    def _preprocess_fn(self, inputs):
        if inputs.ndim == 1:
            inputs = np.expand_dims(inputs, 0)
        outputs = inputs[:, :2*self.block_length].reshape(
            (-1, self.block_length, 2))
        return outputs
