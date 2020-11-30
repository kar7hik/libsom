import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import ghrsom
from timeit import default_timer as timer


rows = 10
cols = 10
raw_data = np.random.randint(0, 255, (1000, 3))
data = raw_data
col_maxes = raw_data.max(axis=0)
data = raw_data / col_maxes[np.newaxis, :]

initial_learning_rate = 0.25
initial_neighbor_radius = 1.5
alpha = 0.7
num_cycle = 5
num_repeat = 2
epochs = 20
num_iteration = 500


dataset_percentage = 0.50
min_size = 5
seed = None

wg = np.random.random((rows, cols, data.shape[1]))

som_normal = ghrsom.SOM(rows,
                        cols,
                        data,
                        initial_learning_rate,
                        initial_neighbor_radius,
                        num_iteration)

som_batch = ghrsom.SOM(rows,
                       cols,
                       data,
                       initial_learning_rate,
                       initial_neighbor_radius,
                       epochs)

rsom_normal = ghrsom.RSOM(rows,
                          cols,
                          data,
                          initial_learning_rate,
                          initial_neighbor_radius,
                          num_iteration)


def plot_data(som_map, rows, cols):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, rows + 1))
    ax.set_ylim((0, cols + 1))

    for x in range(1, rows + 1):
        for y in range(1, cols + 1):
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                           facecolor=som_map[x-1, y-1, :],
                                           edgecolor='none'))
    plt.show()



# som_normal.som_map = wg
# start1 = timer()
# som_normal.som_train()
# end1 = timer()
# print(end1 - start1) # Time in seconds, e.g. 5.38091952400282
# plot_data(som_normal.som_map, rows, cols)


# som_batch.som_map = wg
# start2 = timer()
# som_batch.som_batch_training(dataset_percentage, min_size)
# end2 = timer()
# print(end2 - start2) # Time in seconds, e.g. 5.38091952400282
# plot_data(som_batch.som_map, rows, cols)


rsom_normal.som_map = wg
rsom_normal.rsom_training(num_iteration,
                          5,
                          2,
                          0.7)
plot_data(rsom_normal.som_map, rows, cols)

