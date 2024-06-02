import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import datetime as dt
from tkcalendar import Calendar
from lib.traning_data import TrainingData
from lib.models.BackpropagationModel import BackpropagationModel
from lib.ploter.plot import Ploter
from lib.models import settings

class CryptoGraphicGui:
    """
    This is class for UI on project
    """
    def __init__(self, root):
        """
        This is default constructor
        """
        self.root = root
        self.data = None
        self.algorithm_settings = settings.Settings()
        self.root.title("Crypto and Gold Analysis")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        self.stock = None  # Initialize the stock variable
        self.start_date_calendar = None
        self.end_date_calendar = None

        # Create a frame for the input fields and buttons
        self.input_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.input_frame.pack(pady=10, padx=10, fill=tk.X)

        # Start Date Calendar
        self.start_date_label = tk.Label(self.input_frame, text="Start Date:", bg="#f0f0f0")
        self.start_date_label.grid(row=0, column=0, padx=5, pady=5)
        self.start_date_calendar = Calendar(self.input_frame, selectmode="day", date_pattern="yyyy-mm-dd")
        self.start_date_calendar.grid(row=0, column=1, padx=5, pady=5)

        # End Date Calendar
        self.end_date_label = tk.Label(self.input_frame, text="End Date:", bg="#f0f0f0")
        self.end_date_label.grid(row=0, column=2, padx=5, pady=5)
        self.end_date_calendar = Calendar(self.input_frame, selectmode="day", date_pattern="yyyy-mm-dd")
        self.end_date_calendar.grid(row=0, column=3, padx=5, pady=5)

        # Percentage Split Scale
        self.split_label = tk.Label(self.input_frame, text="Training-Testing Split (%):", bg="#f0f0f0")
        self.split_label.grid(row=1, column=0, padx=5, pady=5)
        self.split_scale = tk.Scale(self.input_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.split_scale.grid(row=1, column=1, padx=5, pady=5)

        # Dropdown menu for selecting crypto or gold
        self.cryptos = ["Bitcoin", "Ethereum", "Ripple", "Litecoin", "Dogecoin", "Gold"]
        self.selected_crypto = tk.StringVar()
        self.selected_crypto.trace('w', self.on_crypto_change)  # Trace changes to the selected crypto
        self.dropdown = ttk.Combobox(self.input_frame, textvariable=self.selected_crypto, values=self.cryptos, state="readonly", width=20)
        self.dropdown.grid(row=1, column=2, padx=5, pady=5)

        # Buttons
        self.train_button = tk.Button(self.input_frame, text="Train", command=self.on_train_button_click, bg="#4CAF50", fg="white", width=12)
        self.train_button.grid(row=2, column=0, padx=5, pady=10)

        self.test_button = tk.Button(self.input_frame, text="Test", command=self.on_test_button_click, bg="#2196F3", fg="white", width=12)
        self.test_button.grid(row=2, column=1, padx=5, pady=10)

        self.grid_search_button = tk.Button(self.input_frame, text="Grid Search", command=self.on_grid_search_click, bg="#FF9800", fg="white", width=12)
        self.grid_search_button.grid(row=2, column=2, padx=5, pady=10)

        self.visualize_button = tk.Button(self.input_frame, text="Visualize", command=self.visualize, bg="#9C27B0", fg="white", width=12)
        self.visualize_button.grid(row=2, column=3, padx=5, pady=10)

        # Grid Search Parameters Input
        self.create_grid_search_inputs()
        
         # Create the label for displaying the best error
        self.best_error_label = tk.Label(self.root, text="", bg="#f0f0f0")
        self.best_error_label.pack(pady=5)
        self.best_hyperparameters_label = tk.Label(self.root, text="", bg="#f0f0f0")
        self.best_hyperparameters_label.pack(pady=5)

        # Log Area
        self.log_area = tk.Text(self.root, height=15, width=90, state="disabled", bg="#e0e0e0")
        self.log_area.pack(pady=20, padx=10)

    def create_grid_search_inputs(self):
        self.grid_search_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.grid_search_frame.pack(pady=10, padx=10, fill=tk.X)

        # Learning Rates
        self.learning_rate_label = tk.Label(self.grid_search_frame, text="Learning Rates:", bg="#f0f0f0")
        self.learning_rate_label.grid(row=0, column=0, padx=5, pady=5)
        self.learning_rate_from_entry = tk.Entry(self.grid_search_frame, width=10)
        self.learning_rate_from_entry.grid(row=0, column=1, padx=5, pady=5)
        self.learning_rate_from_entry.insert(0, "0.01")
        self.learning_rate_to_entry = tk.Entry(self.grid_search_frame, width=10)
        self.learning_rate_to_entry.grid(row=0, column=2, padx=5, pady=5)
        self.learning_rate_to_entry.insert(0, "0.1")

        # Regularization Strengths
        self.regularization_label = tk.Label(self.grid_search_frame, text="Regularization Strengths:", bg="#f0f0f0")
        self.regularization_label.grid(row=1, column=0, padx=5, pady=5)
        self.regularization_from_entry = tk.Entry(self.grid_search_frame, width=10)
        self.regularization_from_entry.grid(row=1, column=1, padx=5, pady=5)
        self.regularization_from_entry.insert(0, "0.001")
        self.regularization_to_entry = tk.Entry(self.grid_search_frame, width=10)
        self.regularization_to_entry.grid(row=1, column=2, padx=5, pady=5)
        self.regularization_to_entry.insert(0, "0.01")

        # Momentums
        self.momentum_label = tk.Label(self.grid_search_frame, text="Momentums:", bg="#f0f0f0")
        self.momentum_label.grid(row=2, column=0, padx=5, pady=5)
        self.momentum_from_entry = tk.Entry(self.grid_search_frame, width=10)
        self.momentum_from_entry.grid(row=2, column=1, padx=5, pady=5)
        self.momentum_from_entry.insert(0, "0.5")
        self.momentum_to_entry = tk.Entry(self.grid_search_frame, width=10)
        self.momentum_to_entry.grid(row=2, column=2, padx=5, pady=5)
        self.momentum_to_entry.insert(0, "0.9")

        # Hidden Layers
        self.hidden_layers_label = tk.Label(self.grid_search_frame, text="Hidden Layers:", bg="#f0f0f0")
        self.hidden_layers_label.grid(row=3, column=0, padx=5, pady=5)
        self.hidden_layers_from_entry = tk.Entry(self.grid_search_frame, width=10)
        self.hidden_layers_from_entry.grid(row=3, column=1, padx=5, pady=5)
        self.hidden_layers_from_entry.insert(0, "1")
        self.hidden_layers_to_entry = tk.Entry(self.grid_search_frame, width=10)
        self.hidden_layers_to_entry.grid(row=3, column=2, padx=5, pady=5)
        self.hidden_layers_to_entry.insert(0, "10")

    def get_grid_search_parameters(self):
        learning_rates = np.linspace(float(self.learning_rate_from_entry.get()), float(self.learning_rate_to_entry.get()), num=5).tolist()
        regularization_strengths = np.linspace(float(self.regularization_from_entry.get()), float(self.regularization_to_entry.get()), num=5).tolist()
        momentums = np.linspace(float(self.momentum_from_entry.get()), float(self.momentum_to_entry.get()), num=5).tolist()
        hidden_layers = np.arange(int(self.hidden_layers_from_entry.get()), int(self.hidden_layers_to_entry.get()) + 1).tolist()

        return hidden_layers, learning_rates, regularization_strengths, momentums


    def open_start_date_calendar(self):
        top = tk.Toplevel(self.root)
        cal = Calendar(top, selectmode="day", date_pattern="yyyy-mm-dd")
        cal.pack(padx=10, pady=10)
        
        def select_date():
            self.start_date_calendar_button.config(text=cal.get_date())
            self.start_date_calendar = cal.get_date()
            top.destroy()

        select_button = tk.Button(top, text="Select Date", command=select_date)
        select_button.pack(pady=5)

    def open_end_date_calendar(self):
        top = tk.Toplevel(self.root)
        cal = Calendar(top, selectmode="day", date_pattern="yyyy-mm-dd")
        cal.pack(padx=10, pady=10)

        def select_date():
            self.end_date_calendar_button.config(text=cal.get_date())
            self.end_date_calendar = cal.get_date()
            top.destroy()

        select_button = tk.Button(top, text="Select Date", command=select_date)
        select_button.pack(pady=5)

    def on_crypto_change(self, *args):
        """
        This method is called when the selected crypto changes.
        """
        self.stock = self.selected_crypto.get()
        self.log(f"Selected stock: {self.stock}")

    def log(self, message):
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)  # Scroll to the end
        self.log_area.config(state="disabled")

    def load_data(self):
        crypto = self.selected_crypto.get()
        start_date = self.start_date_calendar.get_date()  # Get the start date
        end_date = self.end_date_calendar.get_date()    # Get the end date
        split_percentage = self.split_scale.get()       # Get the split percentage
        
        self.log(f"Loading data for {crypto}...")
        
        # Convert start and end dates to datetime objects (You may need to handle date formats)
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d")

        self.log(f"Loading data for {crypto}...")
        train_data = TrainingData(self.stock, start_date, end_date, split_percentage)
        
        train_data.generate()
        train_data.scale_date()
        self.log(f"Data for {crypto} loaded successfully.")
        
        return train_data

    def on_train_button_click(self):
        
        train_data = self.load_data()
        self.data = train_data.data_frame['Close'].values.reshape(-1, 1)

        backpropagation = BackpropagationModel(1, self.algorithm_settings.hidden_layer, 1)

        training_window = tk.Toplevel(self.root)
        training_window.title("Training Progress")

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

        backpropagation.train(train_data.x_train, train_data.y_train, self.algorithm_settings, update_progress, 'train', self.log)

        messagebox.showinfo("Training Complete", "Training has been completed successfully!")

        predictions = backpropagation.forward(train_data.x_train, train_data.y_train)
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
        train_data = self.load_data()

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
        
        predictions = neural_network.forward(train_data.x_test, train_data.y_test)
        denormalized_predictions = predictions * (np.max(self.data) - np.min(self.data)) + np.min(self.data)

        y_test = train_data.y_test * (np.max(self.data) - np.min(self.data)) + np.min(self.data)

        plot_test_data = Ploter()
        plot_test_data.plot_data(train_data, y_test, denormalized_predictions, train_data.split_index, 'test')

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

        train_data = self.load_data()

        #self.algorithm_settings.set_grid_search_parameters(800, [20], [0.008], [0.000006], [0.006])
        hidden_layers, learning_rates, regularization_strengths, momentums = self.get_grid_search_parameters()
        self.algorithm_settings.set_grid_search_parameters(800,hidden_layers, learning_rates, regularization_strengths, momentums)

        def update_progress(epoch):
            progress['value'] = epoch
            progress_window.update_idletasks()

        progress['maximum'] = self.algorithm_settings.epochs
        results = []
        print(self.algorithm_settings.hidden_layers)
        for hidden_size in self.algorithm_settings.hidden_layers:
            print(hidden_size)
            backpropagation = BackpropagationModel(1, hidden_size, 1)
            best_hyperparameters = backpropagation.grid_search(
                train_data.x_train, train_data.y_train, hidden_size, backpropagation,
                self.algorithm_settings, update_progress, self.log
            )
            results.append(best_hyperparameters)

        best_error = float('inf')
        for best_hyperparameters in results:
            for best_hyperparameter in best_hyperparameters:
                if best_hyperparameter['error'] < best_error:
                    best_error = best_hyperparameter['error']
                    best_hyperparameter_result = best_hyperparameter
                    self.algorithm_settings.set_properties(best_hyperparameter['epoch'],
                        20, best_hyperparameter['learning_rate'],
                        best_hyperparameter['regularization_strength'],
                        best_hyperparameter['momentum']
                    )
        # Update UI labels with the best error and hyperparameters
        self.best_error_label.config(text=f"Best Error: {best_error}")
        self.best_hyperparameters_label.config(text=f"Best Hyperparameters: {best_hyperparameter_result}")
        progress_window.destroy()

    def visualize(self):
        crypto = self.selected_crypto.get()
        self.log(f"Visualizing data for {crypto}...")
        # Assuming you have a plot function for visualization
        plt.figure(figsize=(10, 5))
        plt.title(f"{crypto} Data Visualization")
        plt.plot([1, 2, 3, 4, 5], [5, 6, 7, 8, 9])
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()