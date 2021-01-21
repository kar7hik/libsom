import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Import mnist dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from libsom import som_models
from libsom import utils

mnist = fetch_openml('mnist_784',  data_home="~/Research/mnist/")
input_data = mnist.data

# Normalizing the input data - Maximum Gray scale value is 255 
input_data_normalized = input_data / 255.0
targets = mnist.target
Y = targets

train_data, test_data, y_train, y_test = train_test_split(input_data_normalized,
                                                          Y,
                                                          test_size=0.15,
                                                          random_state=42)


# Input parameters: ###########################################################
map_growing_coefficient = 0.1
hierarchical_growing_coefficient = 0.0001
initial_learning_rate = 0.15
initial_neighbor_radius = 1.5
growing_metric = "qe"
epochs = 15
dataset_percentage = 0.5
min_dataset_size = 30
max_iter = 10
num_cycle = 5
num_repeat = 2
alpha = 0.7


### Using PL-GHRSOM model:
# pl_ghrsom = som_models.PL_GHRSOM(train_data,
#                                  map_growing_coefficient,
#                                  hierarchical_growing_coefficient,
#                                  initial_learning_rate,
#                                  initial_neighbor_radius,
#                                  growing_metric)

# zero_neuron = pl_ghrsom.pl_ghrsom_train(epochs,
#                                         dataset_percentage,
#                                         min_dataset_size,
#                                         num_cycle,
#                                         num_repeat,
#                                         alpha,
#                                         max_iter)



result_path = "/home/karthik/Research/libsom/data/"
result_filename = "zero_neuron_mnist.obj"

### Saving the object:
# utils.pickle_object(result_filename, zero_neuron, path=result_path)

### Loading the object from file:
zero_neuron = utils.load_pickle_object(result_filename, path=result_path)


# def show_some_digits(images, targets, sample_size=24, title_text='Digit {}' ):
#     '''
#     Visualize random digits in a grid plot
#     images - array of flatten gidigs [:,784]
#     targets - final labels
#     '''
#     nsamples = sample_size
#     rand_idx = np.random.choice(images.shape[0], nsamples)
#     images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))

#     img = plt.figure(1, figsize=(15, 12), dpi=160)
#     for index, (image, label) in enumerate(images_and_labels):
#         plt.subplot(np.ceil(nsamples/6.0), 6, index + 1)
#         plt.axis('off')
#         # each image is flat, we have to reshape to 2D array 28x28-784
#         plt.imshow(image.reshape(28,28),
#                    cmap=plt.cm.gray_r,
#                    interpolation='nearest')
#         plt.title(title_text.format(label))
#     plt.show()


# show_some_digits(input_data_normalized, targets)
# plt.imshow(zero_neuron.child_map.weight_map[0][0].reshape(28, 28))
# plt.show()
