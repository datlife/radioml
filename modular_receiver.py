import tensorflow  as tf

class ModularReciver(object):
    """Simulate a two parts: Demod and Decoder."""
    def __init__(num_constellations):
        self.num_constellations = num_constellations

    