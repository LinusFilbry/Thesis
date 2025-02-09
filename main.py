from datetime import datetime

from data_construction.accuracy_during_training import accuracy_during_training
from data_construction.final_accuracy_SGD import train_SGD
from data_construction.final_accuracy_table import acc_trained_SFW_MSFW
from data_construction.heatmap import weight_data_from_visualization_model
from data_construction.magnitude_distributions import weight_magnitude_distribution_linear, \
    filter_magnitude_distribution, weight_magnitude_distribution_convolutional
from data_construction.pruned_accuracy import unstructured_global_pruning_conv, unstructured_local_pruning_conv, \
    structured_pruning
from figure_construction.visualize_accuracy_during_training import visualize_accuracy_curve
from figure_construction.visualize_final_accuracy_table import visualize_accuracy_table
from figure_construction.visualize_heatmap import visualize_heatmap
from figure_construction.visualize_magnitude_distributions import visualize_magnitude_distribution
from figure_construction.visualize_pruned_accuracy import visualize_pruning_accuracy_curve


def print_with_time(message):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{current_time}] {message}')


def main():
    choice = input("To reproduce one of the figures, type it in the format \'5.x\', where x is between 1 and 8. \n"
                   "To obtain the accuracy of SGD and SGD with weight decay on the linear and convolutional model, type \'SGD\' instead.\n")

    match choice:
        case '5.1' | '5.2':
            print_with_time('Producing both figures 5.1 and 5.2 (accuracy curves during training).')
            data, testing_epochs = accuracy_during_training()
            visualize_accuracy_curve(data, testing_epochs, False)
            visualize_accuracy_curve(data, testing_epochs, True)
        case '5.3':
            print_with_time('Producing figure 5.3 (accuracy table of SFW and MSFW).')
            data = acc_trained_SFW_MSFW()
            visualize_accuracy_table(data)
        case '5.4':
            print_with_time('Producing figure 5.4 (weight heatmap).')
            layers_dict = weight_data_from_visualization_model()
            visualize_heatmap(layers_dict)
        case '5.5':
            print_with_time('Producing figure 5.5 (weight magnitude distributions).')
            ranges, data, title = weight_magnitude_distribution_linear()
            visualize_magnitude_distribution(ranges, data, title)
            ranges, data, title = weight_magnitude_distribution_convolutional()
            visualize_magnitude_distribution(ranges, data, title)
        case '5.6':
            print_with_time('Producing figure 5.6 (filter magnitude distributions).')
            ranges, data, title = filter_magnitude_distribution()
            visualize_magnitude_distribution(ranges, data, title)
        case '5.7':
            print_with_time('Producing figure 5.7 (global and local unstructured pruning accuracy curves).')
            data, percentages, title = unstructured_global_pruning_conv()
            visualize_pruning_accuracy_curve(data, percentages, title)
            data, percentages, title = unstructured_local_pruning_conv()
            visualize_pruning_accuracy_curve(data, percentages, title)
        case '5.8':
            print_with_time('Producing figure 5.8 (structured pruning accuracy curves).')
            data, percentages, title = structured_pruning()
            visualize_pruning_accuracy_curve(data, percentages, title)
        case 'SGD':
            print_with_time('Obtaining accuracy results of SGD and SGD with weight decay.')
            train_SGD()
        case _:
            print(f'Input \'{choice}\' not recognised. Type \'5.x\', where x is between 1 and 8, or \'SGD\'.')


if __name__ == "__main__":
    main()