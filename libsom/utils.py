import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib
from sklearn.datasets import load_digits


result_path = "/home/karthik/Research/gh-rsom/results/"
data_shape = 8


def plot_single_digit(digit_data):
    digit_data = digit_data.reshape(data_shape,
                                    data_shape)
    
    plt.imshow(digit_data,
              cmap='bone_r',
              interpolation='sinc')
    plt.show()


def gmap_to_matrix(gmap):
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


def plot_digit_map_data(som_map, plot=True, filename="Unknown"):
    plt.imshow(gmap_to_matrix(som_map),
              cmap='bone_r',
              interpolation='sinc')
    plt.show()



def plot_digit_child(e, gmap):
    if e.inaxes is not None:
        coords = (int(e.ydata // data_shape),
                  int(e.xdata // data_shape))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            interactive_digit_plot(neuron.child_map)


def interactive_digit_plot(gmap):
    # _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots()
    ax.imshow(gmap_to_matrix(gmap.weight_map),
              cmap='bone_r',
              interpolation='sinc')
    fig.canvas.mpl_connect('button_press_event',
                           lambda event: plot_digit_child(event, gmap))
    plt.axis('off')
    fig.show()


def save_digit_map_data(som_map, filename):
    plot_digit_map_data(som_map, plot=False, filename=filename)


def plot_color_child(e, gmap):
    if e.inaxes is not None:
        coords = (int(e.xdata),
                  int(e.ydata))
        print(coords)
        # print("Current map shape: {}".format(gmap.current_som_map.shape))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            # print("Child_map shape: {}".format(neuron.child_map.current_som_map.shape))
            interactive_color_plot(neuron.child_map)
            

def interactive_color_plot(gmap):
    som_map = gmap.weight_map
    rows = som_map.shape[0]
    cols = som_map.shape[1]

    print("Current neuron level: {}".format(gmap.level))
    print("Map size: {}".format(som_map.shape))
    # print(som_map)
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
                           lambda event: plot_color_child(event, gmap))    

    fig.show()



def plot_color_map_data(som_map, plot=True, filename="generated_image"):
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



def save_color_map_data(som_map, filename):
    plot_color_map_data(som_map, plot=False, filename=filename)


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
            # plot_color_map_data(neuron.child_map.weight_map)
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



# def find_map_mean(parent_neuron, test_data, result_list = []):
#     """
#     Finds the Euclidean distance between test data and available maps.
#     The map with minimum mean value of Euclidean distance will be selected as the winner. 
#     """
#     winner_map = None

#     parent_neuron_child_map = list(parent_neuron.child_map.neurons.values())
#     parent_neuron_level = parent_neuron.child_map.level
#     parent_neuron_location = parent_neuron.get_location()
    
#     for neuron in parent_neuron_child_map:
#         if neuron.child_map is not None:
#             dist = np.linalg.norm(neuron.child_map.weight_map - test_data,
#                                   axis=2)
#             mean = dist.mean()
#             # plot_color_map_data(neuron.child_map.weight_map)
#             d = {'level': neuron.child_map.level,
#                  'location': neuron.get_location(),
#                  'weight_map': neuron.child_map.weight_map,
#                  'mean': mean}
#             result_list.append(d)
#             find_map_mean(neuron, test_data, result_list)


# def find_best_matching_map(parent_neuron, test_data):
#     result_list = []
#     find_map_mean(parent_neuron, test_data, result_list)
#     mean_list = [item.get('mean') for item in result_list]
    
#     min_mean = np.argmin(np.asarray(mean_list))
#     min_mean_value = mean_list[min_mean]
#     # print(min_mean, min_mean_value)

#     return min_mean, result_list

def get_best_map(parent_neuron, test_data):
    min_mean, result_dict = find_best_matching_map(parent_neuron,
                                                   test_data)
    weight = result_dict[min_mean].get('weight_map')

    return weight


def get_detailed_best_map(parent_neuron, test_data):
    min_mean, result_list = find_best_matching_map(parent_neuron, test_data)
    level = result_list[min_mean].get('level')
    location = result_list[min_mean].get('location')
    weight = result_list[min_mean].get('weight_map')
    mean =  result_list[min_mean].get('mean')
    map_shape = weight.shape
    print("Level: {}, Location: {}, mean: {}, map shape: {}".format(
        level,
        location,
        mean,
        map_shape))
    
    return level, location, weight, mean


def get_best_map_weight_mean(parent_neuron, test_data):
    """
    Returns the mean value of the weight matrix that is close the test data.
    """
    min_mean, result_dict = find_best_matching_map(parent_neuron,
                                                   test_data)
    weight = result_dict[min_mean].get('weight')

    return weight.mean()


def find_som_levels(parent_neuron):
    """
    Prints the levels, locations and map shapes of every map in the
    trained network
    """
    parent_neuron_level = parent_neuron.level
    parent_neuron_location = parent_neuron.get_location()
    parent_neuron_child_map = list(parent_neuron.child_map.neurons.values())
    for neuron in parent_neuron_child_map:
        if neuron.child_map is not None:
            print(neuron.level,
                  neuron.get_location(),
                  neuron.child_map.weight_map.shape)
            find_som_levels(neuron)


### Checking reverse process:
# reverse_audio = librosa.feature.inverse.mfcc_to_audio(mfcc_librosa, sr)
# sf.write('reverse_audio.wav', reverse_audio, sr, subtype='PCM_24')



def test_speech_datapoint(parent_neuron, test_datapoint):
    assert (test_datapoint.ndim == 2), \
        "Data point must be a 2D array!!!"
    m, r = find_best_matching_map(parent_neuron,
                                  test_datapoint)
    level, location, map_result, mean = get_detailed_best_map(parent_neuron,
                                                              test_datapoint)    
    return level, location, map_result, mean


def test_speech_data(parent_neuron, test_data):
    assert (test_data.ndim == 2), \
        "Test data must be a 2D array !!!"

    num_test_data = len(test_data)
    levels = list()
    x_location = list()
    y_location = list()
    weights = list()
    mean_values = list()
    
    for i in range(num_test_data):
        test_datapoint = test_data[i].reshape(1,
                                              test_data.shape[1])
        level, location, weight, mean = test_speech_datapoint(parent_neuron,
                                                              test_datapoint)
        levels.append(level)
        x_location.append(location[0])
        y_location.append(location[1])
        weights.append(weight)
        mean_values.append(mean)
        # mean_values.append(weight.mean())

    levels = np.asarray(levels)
    x_location = np.asarray(x_location)
    y_location = np.asarray(y_location)
    weights = np.asarray(weights)
    mean_values = np.asarray(mean_values)
        
    return levels, x_location, y_location, weights, mean_values


def plot_speech_bmu_locations(levels,
                              x_location,
                              y_location,
                              mean):
    fig = plt.figure(dpi=300)

    ax = fig.add_subplot(111, projection='3d')
    my_cmap = plt.get_cmap('hsv')
    
    ax.scatter3D(x_location,
                 y_location,
                 levels,
                 c=mean,
                 cmap=my_cmap)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.show()
    
