import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import datetime as dt
from tkinter import Checkbutton
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
        This is the default constructor
        """
        self.root = root
        self.data = None
        self.algorithm_settings = settings.Settings()
        self.root.title("Crypto and Gold Analysis")
        self.root.geometry("1000x900")
        self.root.configure(bg="#f0f0f0")
        self.stock = None  # Initialize the stock variable
        self.start_date_calendar = None
        self.end_date_calendar = None

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create a frame for the input fields and buttons
        self.input_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.input_frame.grid(row=0, column=0, pady=10, padx=10, sticky='nsew')
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        for i in range(4):
            self.input_frame.grid_columnconfigure(i, weight=1)
        for i in range(4):
            self.input_frame.grid_rowconfigure(i, weight=1)

        # Start Date Calendar
        self.start_date_label = tk.Label(self.input_frame, text="Start Date:", bg="#f0f0f0")
        self.start_date_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.start_date_calendar = Calendar(self.input_frame, selectmode="day", date_pattern="yyyy-mm-dd")
        self.start_date_calendar.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # End Date Calendar
        self.end_date_label = tk.Label(self.input_frame, text="End Date:", bg="#f0f0f0")
        self.end_date_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.end_date_calendar = Calendar(self.input_frame, selectmode="day", date_pattern="yyyy-mm-dd")
        self.end_date_calendar.grid(row=0, column=3, padx=5, pady=5, sticky='ew')

        # Percentage Split Scale
        self.split_label = tk.Label(self.input_frame, text="Training-Testing Split (%):", bg="#f0f0f0")
        self.split_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.split_scale = tk.Scale(self.input_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.split_scale.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        # Dropdown menu for selecting crypto or gold
        self.choose_label = tk.Label(self.input_frame, text="Choose stock:", bg="#f0f0f0")
        self.choose_label.grid(row=1, column=2, padx=5, pady=5, sticky='w')
        self.cryptos = ["Bitcoin", "Ethereum", "Ripple", "Litecoin", "Dogecoin", "Gold"]
        self.selected_crypto = tk.StringVar()
        self.selected_crypto.trace('w', self.on_crypto_change)  # Trace changes to the selected crypto
        self.dropdown = ttk.Combobox(self.input_frame, textvariable=self.selected_crypto, values=self.cryptos, state="readonly", width=20)
        self.dropdown.grid(row=1, column=3, padx=5, pady=5, sticky='ew')

        # Add two dropdown menus for selecting attributes for training and predicting
        self.attributes = ["Open", "Close", "date", "Low", "High"]
        self.attributes_predicted = ["Open", "Close", "Low", "High"]

        self.predict_future_var = tk.IntVar()
        self.predict_future_var.trace('w', self.on_predict_future_change)
        self.predict_future_checkbox = tk.Checkbutton(self.input_frame, text="Predict in future", variable=self.predict_future_var, onvalue=1, offvalue=0, bg="#f0f0f0")
        self.predict_future_checkbox.grid(row=3, column=3, padx=5, pady=10, sticky='w')

        # Training attribute
        self.train_by_label = tk.Label(self.input_frame, text="Training by:", bg="#f0f0f0")
        self.train_by_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.selected_train_attr = tk.StringVar()
        self.selected_train_attr.trace('w', self.on_train_attr_change)  # Trace changes to the selected training attribute
        self.train_attr_dropdown = ttk.Combobox(self.input_frame, textvariable=self.selected_train_attr, values=self.attributes, state="readonly", width=20)
        self.train_attr_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky='ew')

        # Predicting attribute
        self.predict_by_label = tk.Label(self.input_frame, text="Predict by:", bg="#f0f0f0")
        self.predict_by_label.grid(row=2, column=2, padx=5, pady=5, sticky='w')
        self.selected_predict_attr = tk.StringVar()
        self.selected_predict_attr.trace('w', self.on_predict_attr_change)  # Trace changes to the selected predicting attribute
        self.predict_attr_dropdown = ttk.Combobox(self.input_frame, textvariable=self.selected_predict_attr, values=self.attributes_predicted, state="readonly", width=20)
        self.predict_attr_dropdown.grid(row=2, column=3, padx=5, pady=5, sticky='ew')

        # Buttons
        self.train_button = tk.Button(self.input_frame, text="Train", command=self.on_train_button_click, bg="#4CAF50", fg="white", width=12)
        self.train_button.grid(row=3, column=0, padx=5, pady=10, sticky='ew')

        self.test_button = tk.Button(self.input_frame, text="Test", command=self.on_test_button_click, bg="#2196F3", fg="white", width=12)
        self.test_button.grid(row=3, column=1, padx=5, pady=10, sticky='ew')

        self.grid_search_button = tk.Button(self.input_frame, text="Grid Search", command=self.on_grid_search_click, bg="#FF9800", fg="white", width=12)
        self.grid_search_button.grid(row=3, column=2, padx=5, pady=10, sticky='ew')

        # Grid Search Parameters Input
        self.create_grid_search_inputs()
        
        # Create the label for displaying the best error
        self.best_error_label = tk.Label(self.root, text="", bg="#f0f0f0")
        self.best_error_label.grid(row=2, column=0, pady=5, sticky='ew')
        self.best_hyperparameters_label = tk.Label(self.root, text="", bg="#f0f0f0")
        self.best_hyperparameters_label.grid(row=3, column=0, pady=5, sticky='ew')

        # Log Area
        self.log_area = tk.Text(self.root, height=15, width=90, state="disabled", bg="#e0e0e0")
        self.log_area.grid(row=4, column=0, pady=20, padx=10, sticky='nsew')

    def on_train_attr_change(self, *args):
        # Update the training attribute in algorithm_settings
        self.algorithm_settings.training_by = self.selected_train_attr.get()

    def on_predict_attr_change(self, *args):
        # Update the predicting attribute in algorithm_settings
        self.algorithm_settings.predicted_by = self.selected_predict_attr.get()
    
    def on_predict_future_change(self, *args):
        self.algorithm_settings.predict_future = bool(self.predict_future_var.get())
        print(self.algorithm_settings.predict_future)

    def create_grid_search_inputs(self):
        self.grid_search_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.grid_search_frame.grid(row=1, column=0, pady=10, padx=10, sticky='ew')
        for i in range(4):
            self.grid_search_frame.grid_columnconfigure(i, weight=1)

        self.hidden_layers_from_label = tk.Label(self.grid_search_frame, text="Settings for grid search:", bg="#f0f0f0")
        self.hidden_layers_from_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        self.hidden_layers_from_label = tk.Label(self.grid_search_frame, text="From:", bg="#f0f0f0")
        self.hidden_layers_from_label.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        self.hidden_layers_to_label = tk.Label(self.grid_search_frame, text="To:", bg="#f0f0f0")
        self.hidden_layers_to_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        self.hidden_layers_step_label = tk.Label(self.grid_search_frame, text="Step:", bg="#f0f0f0")
        self.hidden_layers_step_label.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        
        self.learning_rate_label = tk.Label(self.grid_search_frame, text="Learning Rates:", bg="#f0f0f0")
        self.learning_rate_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.learning_rate_from_entry = tk.Entry(self.grid_search_frame, width=10)
        self.learning_rate_from_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        self.learning_rate_from_entry.insert(0,"0.01")
        self.learning_rate_to_entry = tk.Entry(self.grid_search_frame, width=10)
        self.learning_rate_to_entry.grid(row=1, column=2, padx=5, pady=5, sticky='ew')
        self.learning_rate_to_entry.insert(0,"0.1")
        self.learning_rate_step_entry = tk.Entry(self.grid_search_frame, width=10)
        self.learning_rate_step_entry.grid(row=1, column=3, padx=5, pady=5, sticky='ew')
        self.learning_rate_step_entry.insert(0, "10")

        # Regularization Strengths
        self.regularization_label = tk.Label(self.grid_search_frame, text="Regularization Strengths:", bg="#f0f0f0")
        self.regularization_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.regularization_from_entry = tk.Entry(self.grid_search_frame, width=10)
        self.regularization_from_entry.grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        self.regularization_from_entry.insert(0,"0.001")
        self.regularization_to_entry = tk.Entry(self.grid_search_frame, width=10)
        self.regularization_to_entry.grid(row=2, column=2, padx=5, pady=5, sticky='ew')
        self.regularization_to_entry.insert(0,"0.01")
        self.regularization_step_entry = tk.Entry(self.grid_search_frame, width=10)
        self.regularization_step_entry.grid(row=2, column=3, padx=5, pady=5, sticky='ew')
        self.regularization_step_entry.insert(0, "10")

        # Momentums
        self.momentum_label = tk.Label(self.grid_search_frame, text="Momentums:", bg="#f0f0f0")
        self.momentum_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.momentum_from_entry = tk.Entry(self.grid_search_frame, width=10)
        self.momentum_from_entry.grid(row=3, column=1, padx=5, pady=5, sticky='ew')
        self.momentum_from_entry.insert(0,"0.5")
        self.momentum_to_entry = tk.Entry(self.grid_search_frame, width=10)
        self.momentum_to_entry.grid(row=3, column=2, padx=5, pady=5, sticky='ew')
        self.momentum_to_entry.insert(0,"0.9")
        self.momentum_step_entry = tk.Entry(self.grid_search_frame, width=10)
        self.momentum_step_entry.grid(row=3, column=3, padx=5, pady=5, sticky='ew')
        self.momentum_step_entry.insert(0, "10")

        # Hidden Layers
        self.hidden_layers_label = tk.Label(self.grid_search_frame, text="Hidden Layers:", bg="#f0f0f0")
        self.hidden_layers_label.grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.hidden_layers_from_entry = tk.Entry(self.grid_search_frame, width=10)
        self.hidden_layers_from_entry.grid(row=4, column=1, padx=5, pady=5, sticky='ew')
        self.hidden_layers_from_entry.insert(0,"10")
        self.hidden_layers_to_entry = tk.Entry(self.grid_search_frame, width=10)
        self.hidden_layers_to_entry.grid(row=4, column=2, padx=5, pady=5, sticky='ew')
        self.hidden_layers_to_entry.insert(0,"10")
        self.hidden_layers_step_entry = tk.Entry(self.grid_search_frame, width=10)
        self.hidden_layers_step_entry.grid(row=4, column=3, padx=5, pady=5, sticky='ew')
        self.hidden_layers_step_entry.insert(0, "1")

        

    def get_grid_search_parameters(self):
        learning_rates = np.linspace(float(self.learning_rate_from_entry.get()), float(self.learning_rate_to_entry.get()),int(self.learning_rate_step_entry.get())).tolist()
        regularization_strengths = np.linspace(float(self.regularization_from_entry.get()), float(self.regularization_to_entry.get()), int(self.regularization_step_entry.get())).tolist()
        momentums = np.linspace(float(self.momentum_from_entry.get()), float(self.momentum_to_entry.get()),int(self.momentum_step_entry.get())).tolist()
        hidden_layers = np.arange(int(self.hidden_layers_from_entry.get()), int(self.hidden_layers_to_entry.get()) + 1, int(self.hidden_layers_step_entry.get())).tolist()

        return hidden_layers, learning_rates, regularization_strengths, momentums

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
        train_data = TrainingData(self.stock, start_date, end_date, split_percentage, self.algorithm_settings)
        
        train_data.generate()
        train_data.scale_date()
        self.log(f"Data for {crypto} loaded successfully.")
        
        return train_data
    def _check_exeptions ( self, mode ):
        # Check start and end dates
        start_date = self.start_date_calendar.get_date().strip()
        end_date = self.end_date_calendar.get_date().strip()
        split_percentage = self.split_scale.get()

        if not start_date or not end_date:
            messagebox.showerror("Error", "Start and end dates cannot be empty.")
            self.__init__()
        elif start_date == end_date:
            messagebox.showerror("Error", "Start date and end date cannot be the same.")
            self.__init__()
        elif split_percentage == 0 or split_percentage == 100:
            messagebox.showerror("Error", "Split percentage cannot be 0 or 100.")
            self.__init__()
        # Check if a stock has been chosen
        if not self.selected_crypto.get():
            messagebox.showerror("Error", "Please choose a stock.")
            self.__init__()
        # Check if training attribute has been chosen
        if not self.selected_train_attr.get():
            messagebox.showerror("Error", "Please choose a training attribute.")
            self.__init__()
    
        # Check if predicting attribute has been chosen
        if not self.selected_predict_attr.get():
            messagebox.showerror("Error", "Please choose a predicting attribute.")
            self.__init__()

        if self.selected_predict_attr.get() == self.selected_train_attr.get():
            messagebox.showerror("Error", "Cannot choose train is equal to predict.")
            self.__init__()
    
        # Check if all text boxes contain symbols
        entries = [
            self.learning_rate_from_entry,
            self.learning_rate_to_entry,
            self.regularization_from_entry,
            self.regularization_to_entry,
            self.momentum_from_entry,
            self.momentum_to_entry,
            self.hidden_layers_from_entry,
            self.hidden_layers_to_entry
        ]   
        if mode == "grid_search":
            for entry in entries:
                text = entry.get().strip()
                if not text:  # Check if the text is empty or contains only alphanumeric characters
                    messagebox.showerror("Error", "Text boxes cannot be empty or contain only symbols.")
                    self.__init__()
        
    def on_train_button_click(self):
        self._check_exeptions("train")
        train_data = self.load_data()
        self.data = train_data.data_frame[self.algorithm_settings.predicted_by].values.reshape(-1, 1)

        backpropagation = BackpropagationModel(1, self.algorithm_settings.hidden_layer, 1)

        training_window = tk.Toplevel(self.root)
        training_window.title("Training Progress")
        print(self.algorithm_settings.training_by,self.algorithm_settings.predicted_by)
        label = tk.Label(training_window, text="Training in progress...", font=("Helvetica", 14))
        label.pack(pady=10)

        style = ttk.Style()
        style.configure("TProgressbar", thickness=30)
        progress = ttk.Progressbar(training_window, length=300, mode='determinate', style="TProgressbar")
        progress.pack(pady=10)

        def update_progress(epoch, epochs):
            progress['value'] = (epoch / epochs) * 100
            training_window.update_idletasks()

        backpropagation.train(train_data.x_train, train_data.y_train, self.algorithm_settings, update_progress, 'train', self.log)

        messagebox.showinfo("Training Complete", "Training has been completed successfully!")

        predictions = backpropagation.forward(train_data.x_train, train_data.y_train)
        if str(self.algorithm_settings.training_by) != "date":
            min_date = train_data.data_frame[self.algorithm_settings.training_by].values.reshape(-1, 1).min()
            max_date = train_data.data_frame[self.algorithm_settings.training_by].values.reshape(-1, 1).max()
        else:
            min_date = train_data.data_frame.index.values.reshape(-1, 1).min()
            max_date = train_data.data_frame.index.values.reshape(-1, 1).max()
        denormalized_predictions = predictions * (max_date - min_date) + min_date

        y_train = train_data.y_train * (max_date - min_date) + min_date

        plot_test_data = Ploter()
        plot_test_data.plot_data(train_data, y_train, denormalized_predictions, train_data.split_index, 'train', self.algorithm_settings.predict_future)

        final_weights = backpropagation.get_weights()
        np.savez('out/neural_network_weights.npz', *final_weights)

        training_window.destroy()

    def create_date_range(self, start_date, end_date, split_percentage):
        """
        Creates a list of dates starting from start_date to end_date, based on the split percentage,
        and normalizes them between 0 and 1. Returns both the normalized and non-normalized dates.
        """
        # Convert start and end dates to datetime objects
        start_date_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = dt.datetime.strptime(end_date, "%Y-%m-%d")
        print(end_date_dt)
    
        # Calculate the total number of days between start and end dates
        total_days = (end_date_dt - start_date_dt).days  # Exclude the end date
    
        # Calculate the number of dates based on the split percentage
        num_dates = total_days - int(total_days * (split_percentage / 100)) + 1
    
        # Create a list to store the normalized and non-normalized dates
        normalized_dates = []
        all_dates = []
    
        # Increment date by each day until num_dates is reached and normalize them
        current_date = start_date_dt + dt.timedelta(days=int((split_percentage / 100) * total_days))  # Start from the split point
        for i in range(num_dates):
            # Normalize the date to a value between 0 and 1
            normalized_date = (i + int((split_percentage / 100) * total_days)) / total_days
            normalized_dates.append([normalized_date])  # Append the normalized date as a sublist
            all_dates.append([current_date])  # Append the non-normalized date as a sublist
            current_date += dt.timedelta(days=1)  # Increment date by one day
    
        # Convert the lists of normalized and non-normalized dates to NumPy arrays
        normalized_dates_array = np.array(normalized_dates)
        all_dates_array = np.array(all_dates)

        
        # Return a tuple containing both arrays
        return normalized_dates_array, all_dates_array

    def on_test_button_click(self):
        self._check_exeptions("test")
        train_data = self.load_data()

        loaded_weights = np.load('out/neural_network_weights.npz')
        loaded_weights_input_hidden = loaded_weights['arr_0']
        loaded_bias_hidden = loaded_weights['arr_1']
        loaded_weights_hidden_output = loaded_weights['arr_2']
        loaded_bias_output = loaded_weights['arr_3']
        input_size = loaded_weights_input_hidden.shape[0]
        hidden_size = loaded_weights_hidden_output.shape[0]
        output_size = loaded_weights_hidden_output.shape[1]

        
        neural_network = BackpropagationModel(input_size, hidden_size, output_size)
        neural_network.weights_input_hidden = loaded_weights_input_hidden
        neural_network.bias_hidden = loaded_bias_hidden
        neural_network.weights_hidden_output = loaded_weights_hidden_output
        neural_network.bias_output = loaded_bias_output
        self.data = train_data.data_frame[self.algorithm_settings.predicted_by].values.reshape(-1, 1)
        if self.algorithm_settings.predict_future == 1:
            start_date = self.start_date_calendar.get_date()
            end_date = self.end_date_calendar.get_date()
            split_percentage = self.split_scale.get()
            x_test, train_data = self.create_date_range(start_date, end_date, split_percentage)
            predictions = neural_network.forward(x_test, 0)
            denormalized_predictions = predictions * (np.max(self.data) - np.min(self.data)) + np.min(self.data)
            plot_test_data = Ploter()
            plot_test_data.plot_data(train_data, 0, denormalized_predictions, 0, 'test', self.algorithm_settings.predict_future)
            #print(x_test_dernomalizated)
        else:
            x_test = train_data.x_test
            predictions = neural_network.forward(x_test, 0)
            denormalized_predictions = predictions * (np.max(self.data) - np.min(self.data)) + np.min(self.data)
            y_test = train_data.y_test * (np.max(self.data) - np.min(self.data)) + np.min(self.data)
            plot_test_data = Ploter()
            plot_test_data.plot_data(train_data, y_test, denormalized_predictions, train_data.split_index, 'test', self.algorithm_settings.predict_future)

    def on_grid_search_click(self):
        self._check_exeptions("grid_search")
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Grid Search Progress")

        label = tk.Label(progress_window, text="Grid search in progress...", font=("Helvetica", 14))
        label.pack(pady=10)
        print(self.algorithm_settings.learning_rates)
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
                        best_hyperparameter['hidden_layer'], best_hyperparameter['learning_rate'],
                        best_hyperparameter['regularization_strength'],
                        best_hyperparameter['momentum']
                    )
        # Update UI labels with the best error and hyperparameters
        self.best_error_label.config(text=f"Best Error: {best_error}")
        self.best_hyperparameters_label.config(text=f"Best Hyperparameters: {best_hyperparameter_result}")
        progress_window.destroy()