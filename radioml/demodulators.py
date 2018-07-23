import numpy as np
from tensorflow.keras.models import load_model

class Demodulator(object):
    """Abstract Class of Demodulator"""
    def __init__(self):
        pass
    def demodulate(self, inputs):
        pass

class ClassicDemodulator(Demodulator):
    """Classic Demod"""
    def __init__(self, modem):
        super(ClassicDemodulator, self).__init__()
        self.modem = modem
    def demodulate(self, inputs):
        return self.modem.demodulate(inputs, demod_type='hard')                                    


class NeuralDemodulator(Demodulator):
    """Neural-based Demodulation"""
    def __init__(self, model_path, symbol_mapping):
        self.model = load_model(model_path, compile=False)
        self.symbol_mapping = symbol_mapping                                       

    def demodulate(self, inputs):
        predictions = self.model.predict(self._preprocess_fn(inputs))
        symbols = np.argmax(predictions, -1)
        
        # Mapping symbols to actual bits
        return  self.symbol_mapping[symbols].flatten()

    def _preprocess_fn(self, complex_inputs):
        """Encode complex inputs to 2D ndarray inputs"""
        x = np.stack((np.array(complex_inputs).real,
                      np.array(complex_inputs).imag),
                      axis=-1)
        return x.reshape((-1, 2))    
