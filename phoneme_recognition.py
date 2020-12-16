import librosa
import python_speech_features
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import seaborn as sns
from libsom import som_models
from libsom import utils
import numpy as np


map_growing_coefficient = 0.1
hierarchical_growing_coefficient = 0.0001
initial_learning_rate = 0.15
initial_neighbor_radius = 1.5
growing_metric = "qe"
epochs = 20
dataset_percentage = 0.5
min_dataset_size = 10
max_iter = 20
num_cycle = 5
num_repeat = 2
alpha = 0.7


n_mfcc = 33
n_mels = 40
n_fft = 512
hop_length = 512
fmin = 0
fmax = None
# sr = 22050


y, sr = librosa.load("./data/hello.wav")
a, s = librosa.load("./data/hello_1.wav")
b, s = librosa.load("./data/hello_2.wav")
c, s = librosa.load("./data/hello_3.wav")

d = np.append(y, a)
e = np.append(d, b)
f = np.append(e, c)

athena, s = librosa.load("./data/athena.wav")

print(f.shape)

test_y, test_sr = librosa.load("./data/hello_test.wav")

def find_som_levels(parent_neuron):
    parent_neuron_level = parent_neuron.level
    parent_neuron_location = parent_neuron.get_location()
    parent_neuron_child_map = list(parent_neuron.child_map.neurons.values())
    for neuron in parent_neuron_child_map:
        if neuron.child_map is not None:
            print(neuron.level, neuron.get_location(), neuron.child_map.weight_map.shape)
            find_som_levels(neuron)



    

mfcc_librosa = librosa.feature.mfcc(y=f,
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    n_mels=n_mels,
                                    hop_length=hop_length,
                                    fmin=fmin,
                                    fmax=fmax,
                                    htk=False)

mfcc_librosa_test = librosa.feature.mfcc(y=test_y,
                                         sr=test_sr,
                                         n_fft=n_fft,
                                         n_mfcc=n_mfcc,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         fmin=fmin,
                                         fmax=fmax,
                                         htk=False)

mfcc_librosa_a = librosa.feature.mfcc(y=athena,
                                      sr=s,
                                      n_fft=n_fft,
                                      n_mfcc=n_mfcc,
                                      n_mels=n_mels,
                                      hop_length=hop_length,
                                      fmin=fmin,
                                      fmax=fmax,
                                      htk=False)

mfcc_librosa = mfcc_librosa.T
mfcc_librosa_test = mfcc_librosa_test.T
mfcc_librosa_a = mfcc_librosa_a.T

# fig, ax = plt.subplots(figsize=(20, 10))
# sns.heatmap(mfcc_librosa)
# plt.show()

# som = som_models.GHRSOM(mfcc_librosa.T,
#                        map_growing_coefficient,
#                        hierarchical_growing_coefficient,
#                        initial_learning_rate,
#                        initial_neighbor_radius,
#                        growing_metric)

plsom = som_models.PL_GHSOM(mfcc_librosa,
                            map_growing_coefficient,
                            hierarchical_growing_coefficient,
                            initial_learning_rate,
                            initial_neighbor_radius,
                            growing_metric)


zero_neuron = plsom.pl_ghsom_train(epochs,
                                   dataset_percentage,
                                   min_dataset_size,
                                   # num_cycle,
                                   # num_repeat,
                                   # alpha,
                                   max_iter)
# find_som_levels(zero_neuron)


def test_speech_datapoint(parent_neuron, test_datapoint):
    assert (test_datapoint.ndim == 2), \
        "Data point must be a 2D array!!!"
    m, r = utils.find_best_matching_map(parent_neuron,
                                        test_datapoint)
    level, location, map_result, mean = utils.get_best_map(parent_neuron,
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

    levels = np.asarray(levels)
    x_location = np.asarray(x_location)
    y_location = np.asarray(y_location)
    weights = np.asarray(weights)
    mean_values = np.asarray(mean_values)
        
    return levels, x_location, y_location, weights, mean_values


levels, x_location, y_location, weights, mean_values = test_speech_data(zero_neuron,
                                                                        mfcc_librosa_test)


print("Completed")
levels_a, x_location_a, y_location_a, weights_a, mean_a = test_speech_data(zero_neuron,
                                                                           mfcc_librosa)

def plot_speech_bmu_locations(levels,
                              x_location,
                              y_location,
                              mean):
    fig = plt.figure()
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


    
# plot_speech_bmu_locations(levels, x_location, y_location, mean_values)
# plot_speech_bmu_locations(levels_a, x_location_a, y_location_a, mean_a)



# fig = plt.figure()
# ax0 = fig.add_subplot(211, projection='3d')
# ax1 = fig.add_subplot(212, projection='3d')
# my_cmap = plt.get_cmap('hsv')

# ax0.scatter3D(x_location,
#              y_location,
#              levels,
#              c=mean_values,
#              cmap=my_cmap)
# ax1.scatter3D(x_location_a,
#               y_location_a,
#               levels_a,
#               c=mean_a,
#               cmap=my_cmap)

# plt.show()
