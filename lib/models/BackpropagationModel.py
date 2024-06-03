"""
class representing backpropagation with grid search
"""
import numpy as np
from lib.ploter.plot import Ploter
from lib.models import settings

class BackpropagationModel:
    """A class representing a backpropagation neural network model."""

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the neural network model."""
        self.best_hyperparameters = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        np.random.seed(42)  # Set a seed for reproducibility

        # Weight Initialization (Example: He initialization)
        stddev_input_hidden = np.sqrt(2.0 / (input_size + hidden_size))
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) \
                                                        * stddev_input_hidden

        stddev_hidden_output = np.sqrt(2.0 / (hidden_size + output_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) \
                                                        * stddev_hidden_output
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.hidden_activation = None
        self.output = None
        self.output_size = output_size

        # Momentum initialization
        self.momentum = None
        self.velocity_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.velocity_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.velocity_bias_hidden = np.zeros_like(self.bias_hidden)
        self.velocity_bias_output = np.zeros_like(self.bias_output)

    def reset(self):
        """Reset the neural network model."""
        np.random.seed(42)  # Set a seed for reproducibility

        # Weight Initialization (Example: He initialization)
        stddev_input_hidden = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) \
                                                    * stddev_input_hidden

        stddev_hidden_output = np.sqrt(2.0 / (self.hidden_size + self.output_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) \
                                                    * stddev_hidden_output
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        self.hidden_activation = None
        self.output = None

        # Momentum initialization
        self.momentum = None
        self.velocity_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.velocity_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.velocity_bias_hidden = np.zeros_like(self.bias_hidden)
        self.velocity_bias_output = np.zeros_like(self.bias_output)

    def sigmoid(self, x):
        """Apply the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Calculate the derivative of the sigmoid activation function."""
        return x * (1 - x)

    def forward(self, x, y):
        """Perform a forward pass through the network."""
        self.hidden_activation = self.sigmoid(np.dot(x, self.weights_input_hidden) \
                                             + self.bias_hidden)
        self.output = self.sigmoid(np.dot(self.hidden_activation, self.weights_hidden_output) \
                                  + self.bias_output)
        #error = (y - self.output)
        #loss = np.mean(error**2)
        #print(loss)
        return self.output

    def backward(self, x, y, learning_rate, regularization_strength):
        """Perform a backward pass and update weights with momentum."""
        error = y - self.output
        output_delta = error * self.sigmoid_derivative(self.output)

        error_hidden = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = error_hidden * self.sigmoid_derivative(self.hidden_activation)

        # Update velocities
        self.velocity_weights_hidden_output = (self.momentum * \
                                               self.velocity_weights_hidden_output) + (
            self.hidden_activation.T.dot(output_delta) - regularization_strength
            * self.weights_hidden_output) * learning_rate

        self.velocity_bias_output = (self.momentum * self.velocity_bias_output) + \
        np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.velocity_weights_input_hidden = (self.momentum * self.velocity_weights_input_hidden) \
                                + (x.T.dot(hidden_delta) - regularization_strength * \
                                 self.weights_input_hidden) * learning_rate

        self.velocity_bias_hidden = (self.momentum * self.velocity_bias_hidden) + \
                                     np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

        # Update weights with momentum
        self.weights_hidden_output += self.velocity_weights_hidden_output
        self.bias_output += self.velocity_bias_output
        self.weights_input_hidden += self.velocity_weights_input_hidden
        self.bias_hidden += self.velocity_bias_hidden

        return error

    def train(self, x, y, algo_settings, update_progress, mode, log):
        """Train the neural network."""
        loss = None
        best_predictions = None
        best_loss = 10000
        for epoch in range(algo_settings.epochs):
            # Forward and backward pass for each epoch
            if mode != 'grid_search':
                update_progress(epoch, algo_settings.epochs)
            output = self.forward(x,y)
            self.momentum = algo_settings.momentum
            self.backward(x, y, algo_settings.learning_rate, algo_settings.regularization_strength)

            # Print error every 20 epochs
            if epoch % 5 == 0:
                error = (y - output)
                loss = np.mean(error**2)
                # Add regularization term to the loss
                regularization_term = 0.5 * algo_settings.regularization_strength * (
                    np.sum(self.weights_input_hidden**2) + np.sum(self.weights_hidden_output**2))

                loss += regularization_term
                if loss < best_loss:
                    best_loss = loss
                    best_predictions = self.forward(x,y).copy()
                    self.best_hyperparameters = {
                        'epoch': epoch,
                        'hidden_layer': self.hidden_size,
                        'learning_rate': algo_settings.learning_rate,
                        'regularization_strength': algo_settings.regularization_strength,
                        'momentum': algo_settings.momentum,
                        'error': best_loss
                    }
                #print(f"Iteration {epoch}, Error: {loss}")
                log(f"Iteration {epoch}, Error: {loss}")
        return best_predictions, self.best_hyperparameters, best_loss

    def get_weights(self):
        """Get the current weights of the neural network."""
        return self.weights_input_hidden, self.bias_hidden, \
            self.weights_hidden_output, self.bias_output

    @staticmethod
    def grid_search(x_train, y_train, hidden_size, model, algo_settings, update_progress, log):
        """Perform grid search."""
        best_hyperparameters = []
        mse_values = []
        plot_data = Ploter()
        current_iteration = 0
        total_iterations = len(algo_settings.momentums) * len(algo_settings.learning_rates) * \
                            len(algo_settings.regularization_strengths)
        #print(total_iterations)
        step = algo_settings.epochs / total_iterations
        print(algo_settings.momentums, algo_settings.learning_rates, algo_settings.regularization_strengths)
        for momentum in algo_settings.momentums:
            for lr in algo_settings.learning_rates:
                for reg_strength in algo_settings.regularization_strengths:
                    current_iteration += step
                    update_progress(current_iteration)
                    # Initialize and train the model with current hyperparameters
                    example_settings = settings.Settings()
                    example_settings.set_properties(algo_settings.epochs, hidden_size, \
                                                    lr, reg_strength, momentum)
                    best_model, best_hyperparameter, loss = model.train(x_train, y_train, \
                                                            example_settings, update_progress, 'grid_search', log)

                    best_hyperparameters.append(best_hyperparameter)
                    mse_values.append(loss)
                    model.reset()

        #plot_data.plot_x_y(algo_settings.learning_rates, mse_values, 'learning_rates', 'MSE')
        return best_hyperparameters
