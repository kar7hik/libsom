import ghrsom
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import numpy as np
from timeit import default_timer as timer


def plot_child(e, gmap):
    if e.inaxes is not None:
        coords = (int(e.xdata),
                  int(e.ydata))
        print(coords)
        # print("Current map shape: {}".format(gmap.current_som_map.shape))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            # print("Child_map shape: {}".format(neuron.child_map.current_som_map.shape))
            plot_data(neuron.child_map)

def plot_data(gmap):
    som_map = gmap.weight_map
    rows = som_map.shape[0]
    cols = som_map.shape[1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, rows + 1))
    ax.set_ylim((0, cols + 1))

    for x in range(1, rows + 1):
        for y in range(1, cols + 1):
            # print(som_map[x-1, y-1, :])
            ax.add_patch(patches.Rectangle((x-1.0, y-1.0), 1, 1,
                                           facecolor=som_map[x-1, y-1, :],
                                           edgecolor='none'))

    fig.canvas.mpl_connect('button_press_event',
                           lambda event: plot_child(event, gmap))
    
    fig.show()


# Test data: ###################################################################
raw_data = np.random.randint(0, 255, (100, 3))
data = raw_data
col_maxes = raw_data.max(axis=0)
input_dataset = raw_data / col_maxes[np.newaxis, :]

# To get only the blue values:
# for i in range(1000):
#     input_dataset[i][0] = 0.0
#     input_dataset[i][1] = 0.0


# Input parameters: ###########################################################
map_growing_coefficient = 0.01
hierarchical_growing_coefficient = 0.001
initial_learning_rate = 0.25
initial_neighbor_radius = 1.5
growing_metric = "qe"
epochs = 20
dataset_percentage = 0.5
min_dataset_size = 10
max_iter = 20
num_cycle = 5
num_repeat = 2
alpha = 0.7

num_iteration = 100
training_type = "batch"

if training_type == "normal":
    epochs = num_iteration



# GHSOM Initialization: ########################################################
ghsom = ghrsom.GHSOM(input_dataset,
                     map_growing_coefficient,
                     hierarchical_growing_coefficient,
                     initial_learning_rate,
                     initial_neighbor_radius,
                     growing_metric,
                     training_type)

# GHRSOM Initialization: ########################################################
ghrsom = ghrsom.GHRSOM(input_dataset,
                       map_growing_coefficient,
                       hierarchical_growing_coefficient,
                       initial_learning_rate,
                       initial_neighbor_radius,
                       growing_metric,
                       training_type)
 

# Training: ###################################################################
start = timer()
zero_neuron = ghsom.ghsom_train(epochs,
                                dataset_percentage,
                                min_dataset_size,
                                max_iter)
end = timer()
print(end - start)

# start1 = timer()
# zero_neuron_1 = ghrsom.ghrsom_train(epochs,
#                                     num_cycle,
#                                     num_repeat,
#                                     alpha,
#                                     max_iter)
# end1 = timer()
# print(end1 - start1)


plot_data(zero_neuron.child_map)
# plot_data(zero_neuron_1.child_map)

plt.show()





# For Single layer test: ######################################################
# neuron_creator = Neuron_Creator(hierarchical_growing_coefficient,
#                                 "qe")
# zero_layer = neuron_creator.zero_neuron(input_dataset)
# zero_layer.activation(input_dataset)
# weight_map = np.random.uniform(size=(2, 2, input_dataset.shape[1]))
# er = zero_layer.compute_quantization_error()


# first_layer = Growing_SOM((2, 2),
#                             0.001,
#                             weight_map,
#                             input_dataset,
#                             er,
#                             neuron_creator)

# first_layer_1 = Growing_RSOM((2, 2),
#                              0.001,
#                              weight_map,
#                              input_dataset,
#                              er,
#                              neuron_creator)

# def plot_data(current_som_map):
#     rows = current_som_map.shape[0]
#     cols = current_som_map.shape[1]
#     som_map = current_som_map

#     fig = plt.figure()
#     ax = fig.add_subplot(111, aspect='equal')
#     ax.set_xlim((0, rows + 1))
#     ax.set_ylim((0, cols + 1))

#     for x in range(1, rows + 1):
#         for y in range(1, cols + 1):
#             ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
#                                            facecolor=som_map[x-1, y-1, :],
#                                            edgecolor='none'))
#     plt.show()

# first_layer.gsom_train(epochs,
#                        initial_learning_rate,
#                        initial_neighbor_radius,
#                        dataset_percentage,
#                        min_dataset_size,
#                        max_iter)

# first_layer_1.grsom_train(epochs,
#                           initial_learning_rate,
#                           initial_neighbor_radius,
#                           num_cycle,
#                           num_repeat,
#                           alpha,
#                           max_iter)

# plot_data(first_layer.weight_map)
# plot_data(first_layer_1.weight_map)
