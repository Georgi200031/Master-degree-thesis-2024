from lib.traning_data import TrainingData
import numpy as np
from lib.models.Backpropagation import BackpropagationModel
from lib.ploter.plot import Ploter
from lib.models import settings
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class GoldPricePredictionGUI:
    def __init__(self, root):
        self.root = root
        self.data = None
        self.algorithm_settings = settings.Settings()
        self.root.title("GOLD PRICE PREDICTION GUI")
        
        self.create_buttons()

    def create_buttons(self):
        self.gold_button = tk.Button(
            self.root, text="Gold", command=self.on_gold_button_click,
            font=("Helvetica", 16), padx=20, pady=10
        )
        self.gold_button.pack(pady=20)

        self.train_button = tk.Button(
            self.root, text="Train", command=self.on_train_button_click,
            font=("Helvetica", 16), padx=20, pady=10
        )

        self.test_button = tk.Button(
            self.root, text="Test", command=self.on_test_button_click,
            font=("Helvetica", 16), padx=20, pady=10
        )
        self.grid_search_button = tk.Button(
            self.root, text="Grid Search", command=self.on_grid_search_click,
            font=("Helvetica", 16), padx=20, pady=10
        )

    def on_gold_button_click(self):
        self.gold_button.pack_forget()
        self.grid_search_button.pack(pady=10)
        self.train_button.pack(pady=10)
        self.test_button.pack(pady=10)

    def on_grid_search_click(self):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Grid Search Progress")

        label = tk.Label(progress_window, text="Grid search in progress...", font=("Helvetica", 14))
        label.pack(pady=10)

        style = ttk.Style()
        style.configure("TProgressbar", thickness=30)
        progress = ttk.Progressbar(progress_window, length=300, mode='determinate', style="TProgressbar")
        progress.pack(pady=10)

        progress_window.update()

        train_data = TrainingData()
        train_data.generate()
        train_data.scale_date()

        full_interval = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

        self.algorithm_settings.set_grid_search_parameters( 800, [20], [0.008], [0.000006], [0.006] )

        def update_progress(epoch, total_epochs):
            progress['value'] = epoch 
            progress_window.update_idletasks()

        progress['maximum'] = self.algorithm_settings.epochs
        results = []

        for hidden_size in self.algorithm_settings.hidden_layers:
            backpropagation = BackpropagationModel(1, hidden_size, 1)
            best_hyperparameters = backpropagation.grid_search(
                train_data.X_train, train_data.y_train, hidden_size, backpropagation,self.algorithm_settings, update_progress
            )
            results.append(best_hyperparameters)

        best_error = float('inf')
        for best_hyperparameters in results:
            for best_hyperparameter in best_hyperparameters:
                if(best_hyperparameter['error'] < best_error):
                    best_error = best_hyperparameter['error']
                    self.algorithm_settings.set_properties(best_hyperparameter['epoch'],
                        hidden_size, best_hyperparameter['learning_rate'],
                        best_hyperparameter['regularization_strength'], best_hyperparameter['momentum']
                    )
        progress_window.destroy()

    def on_train_button_click(self):
        train_data = TrainingData()
        train_data.generate()
        train_data.scale_date()

        self.data = train_data.data_frame['Close'].values.reshape(-1, 1)

        backpropagation = BackpropagationModel(1, self.algorithm_settings.hidden_layer, 1)

        training_window = tk.Toplevel(self.root)
        training_window.title("Gold price - Training Progress")

        self.create_training_progress_ui(training_window, backpropagation, train_data)

    def create_training_progress_ui(self, window, backpropagation, train_data):
        label = tk.Label(window, text="Training in progress...", font=("Helvetica", 14))
        label.pack(pady=10)

        style = ttk.Style()
        style.configure("TProgressbar", thickness=30)
        progress = ttk.Progressbar(window, length=300, mode='determinate', style="TProgressbar")
        progress.pack(pady=10)

        def update_progress(epoch, epochs):
            progress['value'] = (epoch / epochs) * 100
            window.update_idletasks()

        backpropagation.train(train_data.X_train, train_data.y_train, self.algorithm_settings, update_progress, 'train')

        messagebox.showinfo("Training Complete", "Training has been completed successfully!")

        predictions = backpropagation.forward(train_data.X_train) 
        min_price = np.min(train_data.data_frame['Close'].values.reshape(-1, 1))
        max_price = np.max(train_data.data_frame['Close'].values.reshape(-1, 1))
        denormalized_predictions = predictions * (max_price - min_price) + min_price

        y_train = train_data.y_train * (max_price - min_price) + min_price

        plot_test_data = Ploter()
        plot_test_data.plot_data(train_data, y_train, denormalized_predictions, train_data.split_index, 'train')

        final_weights = backpropagation.get_weights()
        np.savez('out/neural_network_weights.npz', *final_weights)

        window.destroy()

    def on_test_button_click(self):
        train_data = TrainingData()
        train_data.generate()
        train_data.scale_date()

        loaded_weights = np.load('out/neural_network_weights.npz')
        loaded_weights_input_hidden = loaded_weights['arr_0']
        loaded_bias_hidden = loaded_weights['arr_1']
        loaded_weights_hidden_output = loaded_weights['arr_2']
        loaded_bias_output = loaded_weights['arr_3']

        input_size = loaded_weights_input_hidden.shape[0]
        hidden_size = loaded_weights_hidden_output.shape[0]
        output_size = loaded_weights_hidden_output.shape[1]

        self.data = train_data.data_frame['Close'].values.reshape(-1, 1)
        neural_network = BackpropagationModel(input_size, hidden_size, output_size)
        neural_network.weights_input_hidden = loaded_weights_input_hidden
        neural_network.bias_hidden = loaded_bias_hidden
        neural_network.weights_hidden_output = loaded_weights_hidden_output
        neural_network.bias_output = loaded_bias_output

        predictions = neural_network.forward(train_data.X_test)
        denormalized_predictions = predictions * (np.max(self.data) - np.min(self.data)) + np.min(self.data)
        y_test = train_data.y_test * (np.max(self.data) - np.min(self.data)) + np.min(self.data)

        plot_test_data = Ploter()
        plot_test_data.plot_data(train_data, y_test, denormalized_predictions, train_data.split_index, 'test')