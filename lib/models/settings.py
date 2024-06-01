"""
    This is class to store settings for hyperparameters for 
    model and when use gradient search.
    """
class Settings:
    def __init__(self):
        """
        This is constructure when class was created
        """
        self._epochs = 800
        self._hidden_layer = 20
        self._learning_rate = 0.001
        self._regularization_strength = 0.000002
        self._momentum = 0.0009

        self._learning_rates = []
        self._regularization_strengths = []
        self._momentums = []
        self._hidden_layers = []

    def set_properties(self, epochs=None, hidden_layer=None, 
                       learning_rate=None, regularization_strength=None, momentum=None):
        """
        This fuction set hyperparameters for model
        """
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

    def set_grid_search_parameters( self, epochs=None, hiden_layers=None, 
                                    learning_rates=None, regularization_strengths=None, momentums=None ):
        """
        This class set grid srarch parametrs for gradiend searching.
        """
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
        """
        Property for get epochs of model
        """
        return self._epochs

    @property
    def hidden_layer(self):
        """
        Property for get hidden_layer(neurons size) of model
        """
        return self._hidden_layer

    @property
    def learning_rate(self):
        """
        Property for get learning rate of model
        """
        return self._learning_rate

    @property
    def regularization_strength(self):
        """
        Property for get regularization_strength of model
        """
        return self._regularization_strength

    @property
    def momentum(self):
        """
        Property for get momentum of model
        """
        return self._momentum

    @property
    def learning_rates(self):
        """
        Property for get learning_rates for grid search
        """
        return self._learning_rates

    @property
    def hidden_layers(self):
        """
        Property for get hidden_layers(numbers of neurons) for grid search
        """
        return self._hidden_layers

    @property
    def momentums(self):
        """
        Property for get momentums for grid search
        """
        return self._momentums

    @property
    def regularization_strengths(self):
        """
        Property for get regularization_strengths for grid search
        """
        return self._regularization_strengths
