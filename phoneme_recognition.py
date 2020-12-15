import librosa
import python_speech_features
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import seaborn as sns
from libsom import som_models
from libsom import utils
import numpy as np


map_growing_coefficient = 0.01
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
sr = 22050


y, sr = librosa.load("./data/hello.wav")
a, s = librosa.load("./data/hello_1.wav")
b, s = librosa.load("./data/hello_2.wav")
c, s = librosa.load("./data/hello_3.wav")

d = np.append(y, a)
e = np.append(d, b)
f = np.append(e, c)

print(f.shape)

test_y, test_sr = librosa.load("./data/hello_3.wav")

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

mfcc_librosa_test = mfcc_librosa_test.T

# fig, ax = plt.subplots(figsize=(20, 10))
# sns.heatmap(mfcc_librosa)
# plt.show()

som = som_models.GHRSOM(mfcc_librosa.T,
                       map_growing_coefficient,
                       hierarchical_growing_coefficient,
                       initial_learning_rate,
                       initial_neighbor_radius,
                       growing_metric)

zero_neuron = som.ghrsom_train(epochs,
                               num_cycle,
                               num_repeat,
                               alpha,
                               max_iter)
# find_som_levels(zero_neuron)


for i in range(len(mfcc_librosa_test)):
    mfcc_librosa_test_data = mfcc_librosa_test[i].reshape(1, 33)
    m, r = utils.find_best_matching_map(zero_neuron, mfcc_librosa_test_data)
    map_result = utils.get_best_map(zero_neuron, mfcc_librosa_test_data)

    

    
