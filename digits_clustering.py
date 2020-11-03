from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import matplotlib
import ghsom
import numpy as np



data_shape = 8

def __gmap_to_matrix(gmap):
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

def interactive_plot(gmap):
    # _num = "level {} -- parent pos {}".format(level, num)
    fig, ax = plt.subplots()
    ax.imshow(__gmap_to_matrix(gmap.current_som_map),
              cmap='bone_r',
              interpolation='sinc')
    fig.canvas.mpl_connect('button_press_event',
                           lambda event: __plot_child(event, gmap))
    plt.axis('off')
    fig.show()


def __plot_child(e, gmap):
    if e.inaxes is not None:
        coords = (int(e.ydata // data_shape),
                  int(e.xdata // data_shape))
        neuron = gmap.neurons[coords]
        if neuron.child_map is not None:
            interactive_plot(neuron.child_map)


digits = load_digits()
data = digits.data

tau_1 = 0.1
tau_2 = 0.001
learning_rate = 0.95
growing_metric = "qe"

gh = ghsom.GHSOM(data, tau_1, tau_2, learning_rate, growing_metric)
zero_unit = gh.train(50)
gmap = zero_unit.child_map.current_som_map
interactive_plot(zero_unit.child_map)
