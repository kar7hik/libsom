from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import matplotlib
from ghsom import *
import numpy as np



data_shape = 8

def __gmap_to_matrix(gmap):
    map_row = data_shape * gmap.shape[0]
    map_col = data_shape * gmap.shape[1]
    _image = np.empty(shape=(map_row, map_col),
                      dtype=np.float32)
    for i in range(0, map_row, data_shape):
        for j in range(0, map_col, data_shape):
            neuron = gmap[i // data_shape, j // data_shape]
            _image[i:(i + data_shape),
                   j:(j + data_shape)] = np.reshape(neuron,
                                                    newshape=(data_shape,
                                                              data_shape))
    return _image

def interactive_plot(gmap):
    # _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots()
    ax.imshow(__gmap_to_matrix(gmap.weight_map),
              cmap='bone_r',
              interpolation='sinc')
    fig.canvas.mpl_connect('button_press_event',
                           lambda event: __plot_child(event, gmap))
    plt.axis('off')
    fig.show()


def __plot_child(e, gmap):
    if e.inaxes is not None:
        coords = (int(e.ydata // data_shape),
                  int(e.xdata // data_shape))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            interactive_plot(neuron.child_map)


digits = load_digits()
data = digits.data


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

# ghsom = GHSOM(data,
#               map_growing_coefficient,
#               hierarchical_growing_coefficient,
#               initial_learning_rate,
#               initial_neighbor_radius,
#               growing_metric)

# zero_neuron = ghsom.ghsom_train()


ghrsom = GHRSOM(data,
                map_growing_coefficient,
                hierarchical_growing_coefficient,
                initial_learning_rate,
                initial_neighbor_radius,
                growing_metric)

zero_neuron_1 = ghrsom.ghrsom_train()

gmap = zero_neuron_1.child_map.weight_map
interactive_plot(zero_neuron_1.child_map)
