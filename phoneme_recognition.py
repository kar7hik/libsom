import librosa
import python_speech_features
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import seaborn as sns
from ghrsom import *



map_growing_coefficient = 0.01
hierarchical_growing_coefficient = 0.001
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



n_mfcc = 13
n_mels = 40
n_fft = 512
hop_length = 512
fmin = 0
fmax = None
sr = 22050
y, sr = librosa.load("./english.wav")

mfcc_librosa = librosa.feature.mfcc(y=y,
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    n_mels=n_mels,
                                    hop_length=hop_length,
                                    fmin=fmin,
                                    fmax=fmax,
                                    htk=False)


# fig, ax = plt.subplots(figsize=(20, 10))
# sns.heatmap(mfcc_librosa)
# plt.show()

som = GHSOM(mfcc_librosa.T,
            map_growing_coefficient,
            hierarchical_growing_coefficient,
            initial_learning_rate,
            initial_neighbor_radius,
            growing_metric)

zero_neuron = som.ghsom_train()
