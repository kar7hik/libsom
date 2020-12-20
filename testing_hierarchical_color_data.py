import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from libsom import som_models
from sklearn.datasets import load_digits
from libsom import utils
import pickle


input_data = np.random.random((500, 3))
test_data = np.random.random((20, 3))

map_growing_coefficient = 0.01
hierarchical_growing_coefficient = 0.0001
initial_learning_rate = 0.25
initial_neighbor_radius = 1.5
growing_metric = "qe"
training_type = "batch"

epochs = 10
num_cycle = 3
num_repeat = 1
alpha = 0.7
max_iter = 5

dataset_percentage = 0.5
min_dataset_size = 10

### Using GHSOM model:
# ghsom = som_models.GHSOM(input_data,
#                          map_growing_coefficient,
#                          hierarchical_growing_coefficient,
#                          initial_learning_rate,
#                          initial_neighbor_radius,
#                          growing_metric)


# zero_neuron = ghsom.ghsom_train()


result_path = "/home/karthik/Research/libsom/data/"
result_filename = "zero_neuron.obj"
# neuron_file = open(str(result_path+result_filename), 'wb')
# pickle.dump(zero_neuron, neuron_file)


neuron_file_obj = open(str(result_path+result_filename), 'rb')
neuron_obj = pickle.load(neuron_file_obj)
zero_neuron = neuron_obj

### Using GHRSOM model:
# ghrsom = som_models.GHRSOM(input_data,
#                            map_growing_coefficient,
#                            hierarchical_growing_coefficient,
#                            initial_learning_rate,
#                            initial_neighbor_radius,
#                            growing_metric)

# zero_neuron = ghrsom.ghrsom_train(epochs,
#                                  num_cycle,
#                                  num_repeat,
#                                  alpha,
#                                  max_iter)


### Using PL-GHSOM model:
# pl_ghsom = som_models.PL_GHSOM(input_data,
#                                map_growing_coefficient,
#                                hierarchical_growing_coefficient,
#                                initial_learning_rate,
#                                initial_neighbor_radius,
#                                growing_metric)

# zero_neuron = pl_ghsom.pl_ghsom_train(epochs,
#                                       dataset_percentage,
#                                       min_dataset_size,
#                                       max_iter)


### Using PL-GHRSOM model:
# pl_ghrsom = som_models.PL_GHRSOM(input_data,
#                                  map_growing_coefficient,
#                                  hierarchical_growing_coefficient,
#                                  initial_learning_rate,
#                                  initial_neighbor_radius,
#                                  growing_metric)

# zero_neuron = pl_ghrsom.pl_ghrsom_train(epochs,
#                                         dataset_percentage,
#                                         min_dataset_size,
#                                         num_cycle,
#                                         num_repeat,
#                                         alpha,
#                                         max_iter)




### Evaluating color clustering:
# for i in range(len(test_data)):
#     m, r = utils.find_best_matching_map(zero_neuron, test_data[i])
#     map_result = utils.get_best_map(zero_neuron,
#                                     test_data[i])
#     utils.plot_rgb_data(test_data[i])
#     utils.plot_color_map_data(map_result)



### For interactive color plotting:

# utils.interactive_color_plot(zero_neuron.child_map)
# plt.show()

###################################


