import numpy as np

class EchoStateNetwork:
    def __init__(self, input_size, hidden_size, output_size, momentum):
        # Define hyperparameters to search over
        self.hidden_size = hidden_size
        np.random.seed(42)  # Set a seed for reproducibility

        # Weight Initialization (Example: He initialization)
        stddev_input_hidden = np.sqrt(2.0 / (input_size + hidden_size))
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * stddev_input_hidden

        stddev_hidden_output = np.sqrt(2.0 / (hidden_size + output_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * stddev_hidden_output
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.hidden_activation = None
        self.output = None

        # Momentum initialization
        self.momentum = momentum
        self.velocity_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.velocity_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.velocity_bias_hidden = np.zeros_like(self.bias_hidden)
        self.velocity_bias_output = np.zeros_like(self.bias_output)

        # Set other parameters like spectral radius, leaky factor, washout period, etc.
        self.spectral_radius = 0.5
        self.leaky_factor = 0.2
        self.washout_period = 200

    def train(self, X_train, y_train):
        num_samples = X_train.shape[0]
        num_time_steps = X_train.shape[1]

        # Initialize reservoir states with shape (reservoir_size, num_time_steps)
        reservoir_states = np.zeros((self.reservoir_size, num_time_steps))

        for i in range(num_samples):
            # Input vector for the current sample
            x = X_train[i]

            # Initialize reservoir state for the current sample
            reservoir_states[:, 0] = np.tanh(np.dot(self.W_in, x))

            # Compute reservoir states for subsequent time steps
            for t in range(1, num_time_steps):
                reservoir_states[:, t] = (1 - self.leaky_factor) * reservoir_states[:, t - 1] + \
                                          self.leaky_factor * np.tanh(np.dot(self.W_res, reservoir_states[:, t - 1]) +
                                                                      np.dot(self.W_in, x))

        # Compute output weights using ridge regression
        extended_states = np.vstack((np.ones((1, X_train.shape[0])), reservoir_states))
        self.W_out = np.dot(np.dot(y_train.T, extended_states.T), np.linalg.inv(np.dot(extended_states, extended_states.T)))

    def test(self, X_test):
        # Initialize reservoir states for testing
        reservoir_states = np.zeros((self.reservoir_size, X_test.shape[0]))

        # Iterate through test data
        for t in range(X_test.shape[0]):
            # Update reservoir states
            if t > 0:
                reservoir_states[:, t] = (1 - self.leaky_factor) * reservoir_states[:, t - 1] + \
                                          self.leaky_factor * np.tanh(
                                              np.dot(self.W_in, X_test[t]) + np.dot(self.W_reservoir, reservoir_states[:, t - 1]) + self.bias)

        # Compute output using the learned output weights
        extended_states = np.vstack((np.ones((1, X_test.shape[0])), reservoir_states))
        output = np.dot(self.W_out, extended_states)

        return output

    def grid_search(self, X_train, y_train, reservoir_size, spectral_radii, leaky_factors, washout_periods):
        best_loss = float('inf')
        best_hyperparameters = None

        # Perform grid search
        for res_size in reservoir_size:
            for spec_radius in spectral_radii:
                for leaky_factor in leaky_factors:
                    for washout_period in washout_periods:
                        # Set hyperparameters
                        self.reservoir_size = res_size
                        self.spectral_radius = spec_radius
                        self.leaky_factor = leaky_factor
                        self.washout_period = washout_period

                        # Train the network
                        self.train(X_train, y_train)

                        # Test the network
                        output = self.test(X_train)

                        # Compute loss (you need to define your loss function)
                        loss = np.mean((output - y_train) ** 2)

                        # Update best hyperparameters if needed
                        if loss < best_loss:
                            best_loss = loss
                            best_hyperparameters = (res_size, spec_radius, leaky_factor, washout_period)

        return best_hyperparameters
