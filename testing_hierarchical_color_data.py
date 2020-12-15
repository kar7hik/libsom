import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from libsom import som_models
from sklearn.datasets import load_digits
from libsom import utils


input_data = np.random.random((1000, 3))
test_data = np.random.random((20, 3))

map_growing_coefficient = 0.01
hierarchical_growing_coefficient = 0.0001
initial_learning_rate = 0.25
initial_neighbor_radius = 1.5
growing_metric = "qe"
training_type = "batch"


ghsom = som_models.GHSOM(input_data,
                         map_growing_coefficient,
                         hierarchical_growing_coefficient,
                         initial_learning_rate,
                         initial_neighbor_radius,
                         growing_metric)


zero_neuron = ghsom.ghsom_train()


### Evaluating color clustering:
for i in range(len(test_data)):
    m, r = utils.find_best_matching_map(zero_neuron, test_data[i])
    map_result = utils.get_best_map(zero_neuron, test_data[i])
    utils.plot_rgb_data(test_data[i])
    utils.plot_color_map_data(map_result)



### For interactive color plotting:

# utils.interactive_color_plot(zero_neuron.child_map)
# plt.show()

###################################
