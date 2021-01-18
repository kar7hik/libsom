from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import matplotlib
from libsom import som_models
from libsom import utils
import numpy as np
from libsom import plot_utils


digits = load_digits()
data = digits.data

train_data = data[10:, :]
test_data = data[:10,:]


# Input parameters: ###########################################################
map_growing_coefficient = 0.1
hierarchical_growing_coefficient = 0.0001
initial_learning_rate = 0.15
initial_neighbor_radius = 1.5
growing_metric = "qe"
epochs = 15
dataset_percentage = 0.5
min_dataset_size = 30
max_iter = 10
num_cycle = 5
num_repeat = 2
alpha = 0.7


ghsom = som_models.GHSOM(train_data,
                         map_growing_coefficient,
                         hierarchical_growing_coefficient,
                         initial_learning_rate,
                         initial_neighbor_radius,
                         growing_metric)

zero_neuron = ghsom.ghsom_train()

### For interactive digits plotting:
# utils.interactive_digit_plot(zero_neuron.child_map)
# plt.show()


### Evaluating digit clustering:
for i in range(len(test_data)):
    m, r = utils.find_best_matching_map(zero_neuron, test_data[i])
    map_result = utils.get_best_map_weight(zero_neuron, test_data[i])
    plot_utils.plot_single_digit(test_data[i])
    plot_utils.plot_digit_map_data(map_result)
