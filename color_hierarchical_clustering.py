import ghsom
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import numpy as np


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
    som_map = gmap.current_som_map
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


raw_data = np.random.randint(0, 255, (1000, 3))
data = raw_data
col_maxes = raw_data.max(axis=0)
input_dataset = raw_data / col_maxes[np.newaxis, :]

# To get only the blue values:
# for i in range(1000):
#     input_dataset[i][0] = 0.0
#     input_dataset[i][1] = 0.0


tau_1 = 0.1
tau_2 = 0.006
learning_rate = 0.15
growing_metric = "qe"


from timeit import default_timer as timer

gh = ghsom.GHSOM(input_dataset, tau_1, tau_2, learning_rate, growing_metric)
start1 = timer()
zero_unit = gh.train(10)
end1 = timer()
print(end1 - start1)


gmap = zero_unit.child_map.current_som_map
g = zero_unit.child_map
plot_data(g)
plt.show()




# For Single layer test: ######################################################
# neuron_creator = ghsom.Neuron_Creator(0.001, "qe")
# zero_layer = ghsom.Zero_layer(input_dataset, neuron_creator)
# zero_layer.zero_unit.weight_vector
# zero_layer.zero_unit.find_distance(input_dataset)
# er = zero_layer.zero_unit.compute_quantization_error()

# zero_neuron = zero_layer.zero_unit
# weight_map = np.random.uniform(size=(2, 2, input_dataset.shape[1]))
# first_layer = ghsom.Growing_SOM((2, 2),
#                                 0.0037,
#                                 weight_map,
#                                 input_dataset,
#                                 er,
#                                 neuron_creator)

# # first_layer.plot_data()
# first_layer.train(100, 20, 0.95)
# first_layer.plot_data() 
