import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib
from sklearn.datasets import load_digits
from libsom.global_variables import *
from libsom.utils import *
import os
import pickle



def plot_single_digit(digit_data):
    """
    Plots the single digit image - Helpful for test evaluation
    
    ### param:
    digit_data: single image data 
    """
    digit_data = digit_data.reshape(data_shape,
                                    data_shape)
    
    plt.imshow(digit_data,
              cmap='bone_r',
              interpolation='sinc')
    plt.show()


def gmap_to_matrix(gmap):
    """

    """
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
    """
    Function to plot the test data. Uses matplotlib patches to plot the
    rectangle image.

    ### Param:
    rgb_data - Single RGB value.

    """
    fig, ax = plt.subplots()    
    ax.add_patch(patches.Rectangle((0, 0), 10, 10,
                                   facecolor=rgb_data))
    plt.show()


def plot_rgb_data(data):
    """
    Function to plot the RGB data. Takes in 2D array of RBG values.
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]
        
    assert (data.ndim == 2), \
        "Data must be 2D array"
    for i in range(len(data)):
        print(data[i])
        plot_rgb_rectangle(data[i])





def plot_speech_bmu_locations(levels,
                              x_location,
                              y_location,
                              distance_diff_mean):
    fig = plt.figure(dpi=300)

    ax = fig.add_subplot(111, projection='3d')
    my_cmap = plt.get_cmap('hsv')
    
    ax.scatter3D(x_location,
                 y_location,
                 levels,
                 c=distance_diff_mean,
                 cmap=my_cmap)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.show()
