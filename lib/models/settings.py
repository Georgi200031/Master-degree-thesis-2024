class Settings:
    def __init__(self):
        """
        This is constructor when the class is created.
        """
        self._epochs = 800
        self._hidden_layer = 20
        self._learning_rate = 0.001
        self._regularization_strength = 0.000002
        self._momentum = 0.0009
        self.training_by = None
        self.predicted_by = None
        self.predict_future = None

        self._learning_rates = []
        self._regularization_strengths = []
        self._momentums = []
        self._hidden_layers = []

    def set_properties(self, epochs=None, hidden_layer=None,
                       learning_rate=None, regularization_strength=None, momentum=None):
        """
        This function sets hyperparameters for the model.
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

    def set_grid_search_parameters(self, epochs=None, hidden_layers=None,
                                   learning_rates=None, regularization_strengths=None, momentums=None):
        """
        This method sets grid search parameters for gradient searching.
        """
        if learning_rates is not None:
            self._learning_rates = learning_rates
        if regularization_strengths is not None:
            self._regularization_strengths = regularization_strengths
        if momentums is not None:
            self._momentums = momentums
        if epochs is not None:
            self._epochs = epochs
        if hidden_layers is not None:
            self._hidden_layers = hidden_layers

    @property
    def epochs(self):
        """
        Property for getting the epochs of the model.
        """
        return self._epochs

    @property
    def hidden_layer(self):
        """
        Property for getting the hidden_layer(neurons size) of the model.
        """
        return self._hidden_layer

    @property
    def learning_rate(self):
        """
        Property for getting the learning rate of the model.
        """
        return self._learning_rate

    @property
    def regularization_strength(self):
        """
        Property for getting the regularization_strength of the model.
        """
        return self._regularization_strength

    @property
    def momentum(self):
        """
        Property for getting the momentum of the model.
        """
        return self._momentum

    @property
    def learning_rates(self):
        """
        Property for getting the learning_rates for grid search.
        """
        return self._learning_rates

    @property
    def hidden_layers(self):
        """
        Property for getting the hidden_layers(numbers of neurons) for grid search.
        """
        return self._hidden_layers

    @property
    def momentums(self):
        """
        Property for getting the momentums for grid search.
        """
        return self._momentums

    @property
    def regularization_strengths(self):
        """
        Property for getting the regularization_strengths for grid search.
        """
        return self._regularization_strengths
