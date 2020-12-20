import librosa
import soundfile as sf
import python_speech_features
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import seaborn as sns
from libsom import som_models
from libsom import utils
import pickle
import numpy as np


### Parameters for Dataset:
num_people = 4
num_digits = 10

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


n_mfcc = 128
n_mels = 40
n_fft = 1024
hop_length = 1024
fmin = 0
fmax = None
sr = 22050


mfcc_librosa = librosa.feature.mfcc(y=f,
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    n_mels=n_mels,
                                    hop_length=hop_length,
                                    fmin=fmin,
                                    fmax=fmax,
                                    htk=False)


mfcc_librosa = mfcc_librosa.T



som = som_models.GHSOM(mfcc_librosa,
                       map_growing_coefficient,
                       hierarchical_growing_coefficient,
                       initial_learning_rate,
                       initial_neighbor_radius,
                       growing_metric)

zero_neuron = som.ghsom_train()

