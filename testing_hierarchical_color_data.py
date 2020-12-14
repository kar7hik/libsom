import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from ghrsom import refining_ghsom
from sklearn.datasets import load_digits



input_data = np.random.random((1000, 3))

map_growing_coefficient = 0.01
hierarchical_growing_coefficient = 0.0001
initial_learning_rate = 0.25
initial_neighbor_radius = 1.5
growing_metric = "qe"
training_type = "batch"


ghsom = refining_ghsom.GHSOM(input_data,
                             map_growing_coefficient,
                             hierarchical_growing_coefficient,
                             initial_learning_rate,
                             initial_neighbor_radius,
                             growing_metric)


zero_neuron = ghsom.ghsom_train()



def plot_map_data(som_map, plot=True, filename="generated_image"):
    
    rows = som_map.shape[0]
    cols = som_map.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, rows + 1))
    ax.set_ylim((0, cols + 1))

    for x in range(1, rows + 1):
        for y in range(1, cols + 1):
            ax.add_patch(patches.Rectangle((x-1.0, y-1.0), 1, 1,
                                           facecolor=som_map[x-1, y-1, :],
                                           edgecolor='none'))
    if plot is True:
        plt.show()

    else:
        fig.savefig(filename)

def save_map_data(som_map, filename):
    plot_map_data(som_map, plot=False, filename=filename)

result_path = "/home/karthik/Research/gh-rsom/results/"
zero_file_path = result_path + "zero_neuron.png"

## save_map_data(zero_neuron.child_map.weight_map, zero_file_path)


def print_levels(parent_neuron):
    """
    Prints level of each neuron map
    """
    parent_neuron_child_map = list(parent_neuron.child_map.neurons.values())
    parent_neuron_level = parent_neuron.child_map.level
    parent_neuron_location = parent_neuron.get_location()
    
    for neuron in parent_neuron_child_map:
        if neuron.child_map is not None:
            filename = result_path + \
                "parent_level_" + str(parent_neuron_level) + \
                "_parent_location_" + str(parent_neuron_location) + \
                "_level_" + str(neuron.child_map.level) + \
                "_location_" + str(neuron.get_location()) + ".png"

            # print(neuron.child_map.weight_map)
            # plot_map_data(neuron.child_map.weight_map)
            save_map_data(neuron.child_map.weight_map, filename)
            print_levels(neuron)


def plot_rgb_rectangle(rgb_data):
    fig, ax = plt.subplots()    
    ax.add_patch(patches.Rectangle((0, 0), 10, 10,
                                   facecolor=rgb_data))
    plt.show()


def plot_rgb_data(data):
    if data.ndim == 1:
        data = data[np.newaxis, :]
        
    assert (data.ndim == 2), \
        "Data must be 2D array"
    for i in range(len(data)):
        print(data[i])
        plot_rgb_rectangle(data[i])


def find_map_mean(parent_neuron, test_data, result_list = []):
    """
    Finds the Euclidean distance between test data and available maps.
    The map with minimum mean value of Euclidean distance will be selected as the winner. 
    """
    winner_map = None

    parent_neuron_child_map = list(parent_neuron.child_map.neurons.values())
    parent_neuron_level = parent_neuron.child_map.level
    parent_neuron_location = parent_neuron.get_location()
    
    for neuron in parent_neuron_child_map:
        if neuron.child_map is not None:
            dist = np.linalg.norm(neuron.child_map.weight_map - test_data,
                                  axis=2)
            mean = dist.mean()
            #plot_map_data(neuron.child_map.weight_map)
            d = {'level': neuron.child_map.level,
                 'location': neuron.get_location(),
                 'weight_map': neuron.child_map.weight_map,
                 'mean': mean}
            result_list.append(d)
            find_map_mean(neuron, test_data, result_list)


def find_best_matching_map(parent_neuron, test_data):
    result_list = []
    find_map_mean(parent_neuron, test_data, result_list)
    mean_list = [item.get('mean') for item in result_list]
    
    min_mean = np.argmin(np.asarray(mean_list))
    min_mean_value = mean_list[min_mean]
    # print(min_mean, min_mean_value)

    return min_mean, result_list


# def get_specified_map(parent_neuron, level, location, result=None):
#     parent_neuron_child_map = list(parent_neuron.child_map.neurons.values())
#     parent_neuron_level = parent_neuron.child_map.level
#     parent_neuron_location = parent_neuron.get_location()

#     for neuron in parent_neuron_child_map:
#         if neuron.child_map is not None:
#             if not ((neuron.child_map.level == level) and (neuron.get_location() == location)):
#                 pass
#             else:
#                 print(neuron.child_map.level, level, neuron.get_location(), location)
#                 r = neuron.child_map.weight_map
#                 print(r)
#                 result.append(r)

#             get_specified_map(neuron, level, location, result)


def get_best_map(parent_neuron, test_data):
    min_mean, result_list = find_best_matching_map(zero_neuron, test_data)
    level = result_list[min_mean].get('level')
    location = result_list[min_mean].get('location')
    weight = result_list[min_mean].get('weight_map')
    print("Level: {}, Location: {}".format(level, location))
    
    return weight

test_data = np.random.random((20, 3))

for i in range(len(test_data)):
    m, r = find_best_matching_map(zero_neuron, test_data[i])
    map_result = get_best_map(zero_neuron, test_data[i])
    plot_rgb_data(test_data[i])
    plot_map_data(map_result)
