import cog_arch
import numpy as np


nrows = 2
ncols = 2

raw_data = np.random.randint(0, 255, (10, 3))
data = raw_data
col_maxes = raw_data.max(axis=0)
input_dataset = raw_data / col_maxes[np.newaxis, :]


s = cog_arch.SOM(nrows, ncols, input_dataset)
bmus = []
for i in input_dataset:
    bmus.append(s.get_bmu(i))

bmus = np.asarray(bmus)
location = np.asarray(list(np.ndindex(2, 2)))

bb = []
for bmu in bmus:
    bb.append(np.ravel_multi_index(bmu, (2, 2)))

ll = []
for loc in location:
    ll.append(np.ravel_multi_index(loc, (2, 2)))
    


support_stuff = [[position for position in np.where(bb == neuron_idx)[0]]
                 for neuron_idx in ll]


for idx, data_idxs in enumerate(support_stuff):
    print(input_dataset[data_idxs, :])

print(bb)



