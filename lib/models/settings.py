class Settings:
    def __init__(self):
        self._epochs = 100
        self._hidden_layer = 100
        self._learning_rate = 0.6
        self._regularization_strength = 0.01
        self._momentum = 0.001

        self._learning_rates = []
        self._regularization_strengths = []
        self._momentums = []
        self._hidden_layers = []
    
    def set_properties(self, epochs=None, hidden_layer=None, learning_rate=None, regularization_strength=None, momentum=None):
        if hidden_layer is not None:
            self._hidden_layer = hidden_layer
        if learning_rate is not None:
            self._learning_rate = learning_rate
        if regularization_strength is not None:
            self._regularization_strength = regularization_strength
        if momentum is not None:
            self._momentum = momentum
        if epochs is not None:
            self._epochs = epochs
    
    def set_grid_search_parameters( self, epochs=None, hiden_layers=None, learning_rates=None, regularization_strengths=None, momentums=None ):
        if learning_rates is not None:
            self._learning_rates = learning_rates
        if regularization_strengths is not None:
            self._regularization_strengths = regularization_strengths
        if momentums is not None:
            self._momentums = momentums
        if epochs is not None:
            self._epochs = epochs
        if hiden_layers is not None:
            self._hidden_layers = hiden_layers

    @property
    def epochs(self):
        return self._epochs
    
    @property
    def hidden_layer(self):
        return self._hidden_layer
    
    @property
    def learning_rate(self):
        return self._learning_rate
    
    @property
    def regularization_strength(self):
        return self._regularization_strength
    
    @property
    def momentum(self):
        return self._momentum
    
    @property
    def learning_rates(self):
        return self._learning_rates
    
    @property
    def hidden_layers(self):
        return self._hidden_layers
    
    @property
    def momentums(self):
        return self._momentums
    
    @property
    def regularization_strengths(self):
        return self._regularization_strengths