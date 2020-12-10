import numpy as np
import refining_ghsom
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches as patches
matplotlib.use('TKAgg')


raw_data = np.random.randint(0, 255, (100, 3))
data = raw_data
col_maxes = raw_data.max(axis=0)
input_data = raw_data / col_maxes[np.newaxis, :]

map_growing_coefficient = 0.01
hierarchical_growing_coefficient = 0.001
initial_learning_rate = 0.25
initial_neighbor_radius = 1.5
growing_metric = "qe"
training_type = "batch"


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

epochs = 15
dataset_percentage = 0.5
min_dataset_size = 60
max_iter = 200

ghsom = refining_ghsom.GHSOM(input_data,
                             map_growing_coefficient,
                             hierarchical_growing_coefficient,
                             initial_learning_rate,
                             initial_neighbor_radius,
                             growing_metric,
                             training_type)


zero_neuron = ghsom.ghsom_train(epochs,
                                dataset_percentage,
                                min_dataset_size,
                                max_iter)

plot_data(zero_neuron.child_map)
plt.show()
