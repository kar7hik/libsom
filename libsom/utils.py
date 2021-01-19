import numpy as np
from sklearn.datasets import load_digits
from libsom.global_variables import *
from libsom import plot_utils
import os
import pickle



### Serialization:
def pickle_object(filename, obj, path="./"):
    """
    Function to pickle an object.

    ### Params:
    filename - Name of the file to be stored
    obj - Object to be pickled
    path - default path
    """
    filepath = os.path.join(path, filename)
    with open(filepath, 'wb') as obj_file:
        pickle.dump(obj, obj_file)


def load_pickle_object(filename, path="./"):
    """
    Function to load an object from a file. Returns an object.

    ### Params:
    filename - Name of the file to be loaded
    path - default path
    """    
    filepath = os.path.join(path, filename)
    obj = None
    with open(filepath, 'rb') as obj_file:
        obj = pickle.load(obj_file)

    return obj


def find_map_mean(parent_neuron, test_data, result_list = []):
    """
    Finds the Euclidean distance between test data and available maps.
    
    The map with minimum mean value of Euclidean distance will be 
    selected as the winner. 
    """
    winner_map = None

    parent_neuron_child_map = list(parent_neuron.child_map.neurons.values())
    parent_neuron_level = parent_neuron.child_map.level
    parent_neuron_location = parent_neuron.get_location()
    
    for neuron in parent_neuron_child_map:
        if neuron.child_map is not None:
            dist = np.linalg.norm(neuron.child_map.weight_map - test_data,
                                  axis=2)
            dist_diff_mean = dist.mean()
            # plot_color_map_data(neuron.child_map.weight_map)
            d = {'level': neuron.child_map.level,
                 'location': neuron.get_location(),
                 'weight_map': neuron.child_map.weight_map,
                 'dist_diff_mean': dist_diff_mean}
            result_list.append(d)
            find_map_mean(neuron, test_data, result_list)


def find_best_matching_map(parent_neuron, test_data):
    """
    High-level function uses find_map_mean to return the best map for the test data

    Returns min_mean index and a list containing dict of parameters of best maps for
    the test data
    """
    result_list = []
    find_map_mean(parent_neuron, test_data, result_list)
    dist_diff_mean_list = [item.get('dist_diff_mean') for item in result_list]
    
    min_mean = np.argmin(np.asarray(dist_diff_mean_list))
    min_mean_value = dist_diff_mean_list[min_mean]
    # print(min_mean, min_mean_value)

    return min_mean, result_list


def get_best_map_weight(parent_neuron, test_data):
    """
    Function to get the best map weight for the given test data.
    """
    min_mean, result_dict = find_best_matching_map(parent_neuron,
                                                   test_data)
    weight = result_dict[min_mean].get('weight_map')

    return weight


def get_detailed_best_map(parent_neuron, test_data):
    """
    Another function to get the best map for the test data.

    But this function returns multiple details about the best map.
    """
    
    min_mean, result_list = find_best_matching_map(parent_neuron, test_data)
    level = result_list[min_mean].get('level')
    location = result_list[min_mean].get('location')
    weight = result_list[min_mean].get('weight_map')
    dist_diff_mean =  result_list[min_mean].get('dist_diff_mean')
    map_shape = weight.shape
    print("Level: {}, Location: {}, distance_diff_mean: {}, map shape: {}".format(
        level,
        location,
        dist_diff_mean,
        map_shape))
    
    return level, location, weight, dist_diff_mean


def get_best_map_weight_mean(parent_neuron, test_data):
    """
    Returns the mean value of the weight matrix that is close the test data.
    """
    weight = get_best_map_weight(parent_neuron, test_data)
    
    return weight.mean()


def find_som_levels(parent_neuron, plot=False):
    """
    Prints the levels, locations and map shapes of every map in the
    trained network
    """
    parent_neuron_level = parent_neuron.level
    parent_neuron_location = parent_neuron.get_location()
    parent_neuron_child_map = list(parent_neuron.child_map.neurons.values())
    for neuron in parent_neuron_child_map:
        if neuron.child_map is not None:
            if plot:
                filename = str(neuron.level) + "_" + str(neuron.get_location())
                filepath = som_layer_result_path + filename
                plot_utils.save_color_map_data(neuron.child_map.weight_map,
                                               filepath)
                print(neuron.level,
                      neuron.get_location(),
                      neuron.child_map.weight_map.shape)

            else:
                print(neuron.level,
                      neuron.get_location(),
                      neuron.child_map.weight_map.shape)
            find_som_levels(neuron, plot=plot)


### Checking reverse process:
# reverse_audio = librosa.feature.inverse.mfcc_to_audio(mfcc_librosa, sr)
# sf.write('reverse_audio.wav', reverse_audio, sr, subtype='PCM_24')



def test_speech_datapoint(parent_neuron, test_datapoint):
    """
    Low-level function that returns the details of the best map for the test_datapoint
    """
    assert (test_datapoint.ndim == 2), \
        "Data point must be a 2D array!!!"
    level, loc, weight, dist_diff_mean = get_detailed_best_map(parent_neuron,
                                                               test_datapoint)    
    return level, loc, weight, dist_diff_mean


def test_speech_data(parent_neuron, test_data):
    """
    High-level function to find the best map for the given test data.
    """
    assert (test_data.ndim == 2), \
        "Test data must be a 2D array !!!"

    num_test_data = len(test_data)
    levels = list()
    x_location = list()
    y_location = list()
    weights = list()
    dist_diff_mean_values = list()
    
    for i in range(num_test_data):
        test_datapoint = test_data[i].reshape(1,
                                              test_data.shape[1])
        level, location, weight, dist_diff_mean = test_speech_datapoint(parent_neuron,
                                                                        test_datapoint)
        levels.append(level)
        x_location.append(location[0])
        y_location.append(location[1])
        weights.append(weight)
        dist_diff_mean_values.append(dist_diff_mean)
        # mean_values.append(weight.mean())

    levels = np.asarray(levels)
    x_location = np.asarray(x_location)
    y_location = np.asarray(y_location)
    weights = np.asarray(weights)
    dist_diff_mean_values = np.asarray(dist_diff_mean_values)
        
    return levels, x_location, y_location, weights, dist_diff_mean_values
