import librosa
import soundfile as sf
import python_speech_features
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import seaborn as sns
from libsom import som_models
from libsom import utils
from libsom.global_variables import *
from libsom import plot_utils
import pickle
import numpy as np
import os
from timeit import default_timer as timer



### Parameters for Dataset:
num_digits = 10

map_growing_coefficient = 0.01
hierarchical_growing_coefficient = 0.0001
initial_learning_rate = 0.15
initial_neighbor_radius = 1.5
growing_metric = "qe"
epochs = 70
dataset_percentage = 0.7
min_dataset_size = 5
max_iter = 20
num_cycle = 5
num_repeat = 2
alpha = 0.7


n_mfcc = 33
n_fft = 1024
hop_length = 1024
fmin = 0
fmax = None
sr = 22050



def wav2mfcc(file_path, max_pad_len=25):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    pad_width = max_pad_len - mfcc.shape[1]
    # print(file_path)
    # print(mfcc.shape)
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc


def wav_info_extractor(dirpath, ext):
    path = dirpath
    mfcc_vectors = []
    
    for file in os.listdir(path):
        if file.endswith(ext):
            file_path = os.path.join(path, file)
            
            mfcc = wav2mfcc(file_path=file_path)
            mfcc_vectors.append(mfcc.T)
            
    features = np.array(mfcc_vectors)
    features = np.concatenate(features, axis=0)
    return features


# features = wav_info_extractor(dirpath=data_dir, ext=".wav")


### Saving features
feature_file = "features.npy"
# np.save(os.path.join(data_path, feature_file), features)


### Loading NPY file:
features = np.load(os.path.join(data_path, feature_file))


som = som_models.GHSOM(features,
                       map_growing_coefficient,
                       hierarchical_growing_coefficient,
                       initial_learning_rate,
                       initial_neighbor_radius,
                       growing_metric)


# start_time = timer() 
# zero_neuron = som.ghsom_train(epochs,
#                               dataset_percentage,
#                               min_dataset_size,
#                               max_iter)
# end_time = timer()
# time_1 = end_time - start_time

# print(time_1)



### Saving the Neuron object:
result_filename = "neuron.obj"
# utils.pickle_object(result_filename, zero_neuron, path=result_path)


### Loading the Neuron object:
zero_neuron = utils.load_pickle_object(result_filename, path=result_path)





# test_audio_file = "0_george_0.wav"
test_audio_file = "one.wav"
test_audio_file_path = os.path.join(data_path, test_audio_file)
test_audio_data, sr = librosa.load(test_audio_file_path)

test_mfcc = librosa.feature.mfcc(y=test_audio_data,
                                 sr=sr,
                                 n_mfcc=n_mfcc,
                                 n_fft=n_fft)

test_mfcc = test_mfcc.T


level, loc_x, loc_y, weight, mean = utils.test_speech_data(zero_neuron,
                                                           test_mfcc)

plot_utils.plot_speech_bmu_locations(level,
                                    loc_x,
                                    loc_y,
                                    mean)


# import plotly.figure_factory as ff
# import plotly.express as px
# z = zero_neuron.child_map.weight_map

# for i in range(len(z)):
#     x = z[i]
#     # fig = ff.create_annotated_heatmap(x)
#     fig = px.imshow(x)
#     fig.show()


# fig = ff.create_dendrogram(z[1])
# fig.show()
