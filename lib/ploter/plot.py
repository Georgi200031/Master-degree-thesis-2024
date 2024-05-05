import matplotlib
import matplotlib.pyplot as plt
class Ploter:
    def __init__( self ):
        matplotlib.use('TkAgg')

    def plot_data( self, data_generator, y_test, denormalized_predictions, split_index, mode ):
        generated_data = data_generator
        if mode == 'test':
            plt.plot(generated_data.data_frame.index[split_index:-1], y_test, label='True Data', color='blue')
            plt.plot(generated_data.data_frame.index[split_index:-1], denormalized_predictions, label='Predictions', color='red')
        if mode == 'train':
            plt.plot(generated_data.data_frame.index[:split_index], y_test, label='True Data', color='blue')
            plt.plot(generated_data.data_frame.index[:split_index], denormalized_predictions, label='Predictions', color='red')
        plt.title('Neural Network Predictions vs True Data, learning rate = 0.008')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    def plot_predicted_x_y_learning_rate( self, x, y, string_name1, string_name2, learning_rate ):
        plt.figure()
        plt.plot(x, label=string_name1)
        plt.plot(y, label=string_name2)
        plt.xlabel(string_name1)
        plt.ylabel(string_name2)
        plt.title(f'{string_name1} vs {string_name2} for lr={learning_rate}')
        #plt.legend()
        plt.show()
    def plot_x_y( self, x, y, string_name1, string_name2):
        plt.figure()
        plt.plot(x, y, label=f'{string_name1} vs {string_name2}')
        plt.xlabel(string_name1)
        plt.ylabel(string_name2)
        plt.title(f'{string_name1} vs {string_name2}')
        plt.legend()
        plt.show()