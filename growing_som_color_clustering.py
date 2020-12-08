import ghrsom
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import numpy as np
from timeit import default_timer as timer


raw_data = np.random.randint(0, 255, (1000, 3))
data = raw_data
col_maxes = raw_data.max(axis=0)
input_data = raw_data / col_maxes[np.newaxis, :]


map_growing_coefficient = 0.01
hierarchical_growing_coefficient = 0.001
growing_metric = "qe"
epochs = 20
num_cycle = 5
num_repeat = 2
num_iteration = 5000
max_iter = 10
alpha = 0.7


initial_learning_rate = 0.95
initial_neighbor_radius = 2
dataset_percentage = 0.50
min_size = 5
seed = None

neuron_creator = ghrsom.Neuron_Creator(hierarchical_growing_coefficient,
                                       "qe")
zero_layer = neuron_creator.zero_neuron(input_data)
weight_map = np.random.uniform(size=(2, 2, input_data.shape[1]))
er = zero_layer.compute_quantization_error()


def plot_data(current_som_map):
    rows = current_som_map.shape[0]
    cols = current_som_map.shape[1]
    som_map = current_som_map

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, rows + 1))
    ax.set_ylim((0, cols + 1))

    for x in range(1, rows + 1):
        for y in range(1, cols + 1):
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                           facecolor=som_map[x-1, y-1, :],
                                           edgecolor='none'))
    plt.show()


gplsom = ghrsom.Growing_PLSOM((2, 2),
                              map_growing_coefficient,
                              weight_map,
                              input_data,
                              er,
                              neuron_creator)

# plot_data(gplsom.weight_map)
gplsom.plsom_train(epochs,
                   dataset_percentage,
                   min_size,
                   training_type="batch")
plot_data(gplsom.weight_map)



# gplrsom = ghrsom.Growing_PLRSOM((2, 2),
#                                 map_growing_coefficient,
#                                 weight_map,
#                                 input_data,
#                                 er,
#                                 neuron_creator)

# plot_data(gplrsom.weight_map)
# gplrsom.plrsom_train(num_iteration,
#                     dataset_percentage,
#                     min_size,
#                     num_cycle,
#                     num_repeat,
#                     alpha,
#                     initial_neighbor_radius,
#                     training_type="normal",
#                     max_iter=max_iter)

# plot_data(gplrsom.weight_map)
