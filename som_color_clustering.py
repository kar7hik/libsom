import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patches
from ghsom import self_organizing
from timeit import default_timer as timer


rows = 10
cols = 10
raw_data = np.random.randint(0, 255, (1000, 3))
data = raw_data
col_maxes = raw_data.max(axis=0)
data = raw_data / col_maxes[np.newaxis, :]
num_iteration = 15

s = self_organizing.SOM(nrows=rows,
                 ncols=cols,
                 input_data=data,
                 num_iteration=num_iteration,
                 init_radius=1.5)

ss = self_organizing.SOM(nrows=rows,
                  ncols=cols,
                  input_data=data,
                  num_iteration=num_iteration,
                  init_radius=1.5)
# r = cog_arch.RSOM(nrows=rows,
#                   ncols=cols,
#                   input_data=data,
#                   num_iteration=num_iteration,
#                   init_learning_rate=0.9,
#                   alpha=0.7,
#                   num_cycle=2,
#                   num_repeat=4)


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



plot_data(s.som_map, rows, cols)

start1 = timer()
s.train_som()
end1 = timer()
print(end1 - start1) # Time in seconds, e.g. 5.38091952400282
plot_data(s.som_map, rows, cols)


dataset_percentage = 0.20
min_size = 30
seed = None
learning_rate = 0.25
neighbor_radius = 1.5

start2 = timer()
ss.batch_train_som(dataset_percentage, min_size, seed)
end2 = timer()
print(end2 - start2) # Time in seconds, e.g. 5.38091952400282
plot_data(ss.som_map, rows, cols)  

# plot_data(r.som_map, rows, cols)
# r.train_recurrent_som_continuous_update()
# plot_data(r.som_map, rows, cols)

