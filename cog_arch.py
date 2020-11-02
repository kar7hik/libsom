import numpy as np
import itertools
from tqdm import tqdm


class SOM():
    def __init__(self,
                 nrows,
                 ncols,
                 input_data,
                 num_iteration=100,
                 init_learning_rate=0.9,
                 init_radius=None):
        self.nrows = nrows
        self.ncols = ncols
        self.input_data = input_data
        self.input_len = len(input_data)
        self.num_input_features = input_data.shape[1]
        self.som_map = np.random.uniform(low=0,
                                         high=1,
                                         size=(self.nrows,
                                               self.ncols,
                                               self.num_input_features))
        self.num_iteration = num_iteration
        self.node_list_ = np.array(list(
            itertools.product(range(self.nrows), range(self.ncols))),
            dtype=int)
        self.init_learning_rate = init_learning_rate
        if init_radius is None:
            self.init_radius = max(self.nrows, self.ncols) / 2
        else:
            self.init_radius = init_radius

        assert self.init_radius != 0, \
            "Init radius cannot be zero."
        self.time_constant = self.num_iteration / np.log(self.init_radius)
        self.umatrix = np.zeros(
            shape=(self.nrows*2-1, self.ncols*2-1, 1),
            dtype=float)

        print("Init radius: {}, num_iteration: {}, time_constant: {}".format(
            self.init_radius,
            self.num_iteration,
            self.time_constant))

    def get_unit_weight(self, location):
        """Get the weights associated with a given unit.
        """
        x_loc = location[0]
        y_loc = location[1]
        return self.som_map[x_loc, y_loc, :]

    def set_unit_weights(self, weights_vector, location):
        """Set the weights associated with a given unit.
        """
        x_loc = location[0]
        y_loc = location[1]
        self.som_map[x_loc, y_loc, :] = weights_vector

    def set_weight_matrix(self, weight_matrix):
        self.som_map = weight_matrix

    def random_input_index(self):
        return np.random.randint(low=0, high=self.input_len)

    def decay_radius(self, current_iter):
        return self.init_radius * np.exp(-current_iter / self.time_constant)

    def decay_learning_rate(self, current_iter):
        return self.init_learning_rate * \
            np.exp(-current_iter / self.num_iteration)

    def calculate_influence(self, w_dist, neighbor_radius):
        pseudogaussian = np.exp(-np.divide(np.power(w_dist, 2),
                                           (2 * np.power(neighbor_radius, 2))))
        return pseudogaussian

    def calculate_influence_old(self, distance, radius):
        return np.exp(-distance / (2 * (radius**2)))

    def get_bmu(self, datapoint):
        """Get best matching unit (BMU) for datapoint.
        Parameters
        ----------
        datapoint : np.array, shape=shape[1]
            Datapoint = one row of the dataset X
        Returns
        -------
        tuple, shape = (int, int)
            Position of best matching unit (row, column)
        """
        a = np.linalg.norm(self.som_map - datapoint, axis=2)
        return np.argwhere(a == np.min(a))[0]

    def modify_weight_matrix(self, learning_rate,
                             neighbor_influence, datapoint):
        """Modify weight matrix of the SOM for the online algorithm.
        Returns
        -------
        np.array
        Weight vector of the SOM after the modification
        """
        return self.som_map + np.multiply(learning_rate, np.multiply(
            neighbor_influence, -np.subtract(self.som_map, datapoint)))

    def som_update(self, current_iter):
        random_index = self.random_input_index()
        bmu_index = self.get_bmu(self.input_data[random_index])
        neighbor_radius = self.decay_radius(current_iter)
        learning_rate = self.decay_learning_rate(current_iter)
        w_dist = np.linalg.norm(self.node_list_ - bmu_index, axis=1)
        neighbor_influence = self.calculate_influence(w_dist,
                                                      neighbor_radius).reshape(
                                                          self.nrows,
                                                          self.ncols,
                                                          1)
        self.som_map = self.modify_weight_matrix(learning_rate,
                                                 neighbor_influence,
                                                 self.input_data[random_index])
        # print(self.som_map)

    def train_som(self):
        for i in tqdm(range(self.num_iteration)):
            self.som_update(i)


class RSOM(SOM):
    def __init__(self,
                 nrows,
                 ncols,
                 input_data,
                 num_iteration=100,
                 init_learning_rate=0.9,
                 init_radius=None,
                 alpha=0.7,
                 num_cycle=4,
                 num_repeat=5):
        SOM.__init__(self,
                     nrows=nrows,
                     ncols=ncols,
                     input_data=input_data,
                     num_iteration=num_iteration,
                     init_learning_rate=init_learning_rate,
                     init_radius=init_radius)
        assert (num_cycle >= 1), \
            "Number of cycle must be greater than zero."
        assert (isinstance(num_cycle, int)), \
            "The number of cycle must be an integer."
        assert (num_repeat >= 1), \
            "Number of repeat must be greater than zero."
        assert (isinstance(num_repeat, int)), \
            "The number of repeat must be an integer."
        assert (alpha <= 1), \
            "Alpha should be less than or equal to 1."
        assert (alpha > 0), \
            "Alpha should be greater than 0."

        self.alpha = alpha
        self.differences = np.zeros(self.som_map.shape)
        self.bmus = []
        self.num_cycle = num_cycle
        self.num_repeat = num_repeat

    def get_bmu(self, datapoint):
        self.differences = self.find_difference_vector(datapoint)
        a = np.linalg.norm(self.differences, axis=2)
        # print(a)
        # print(r.som_map(np.argwhere(a == np.min(a))[0]))
        return np.argwhere(a == np.min(a))[0]

    def find_difference_vector(self, datapoint):
        """
        Parameters:
        y_t: Y at time t
        x_t: X at time t
        w_t: weight value at t
        alpha: Weighting factor determining the effect of
            the difference vector
        """
        # print(self.differences)
        first = ((1 - self.alpha) * self.differences)
        second = (self.alpha * (datapoint - self.som_map))
        y_t = first + second
        # print(y_t)
        return y_t

    def modify_weight_matrix(self, learning_rate, neighbor_influence):
        return self.som_map + np.multiply(learning_rate,
                                          np.multiply(neighbor_influence,
                                                      self.differences))

    def rsom_update(self, current_iter, datapoint, update_map=True):
        r_bmu_index = self.get_bmu(datapoint)
        # print("current iter: {0}".format(current_iter))
        neighbor_radius = self.decay_radius(current_iter)
        learning_rate = self.decay_learning_rate(current_iter)
        w_dist = np.linalg.norm(self.node_list_ - r_bmu_index, axis=1)
        neighbor_influence = self.calculate_influence(w_dist,
                                                      neighbor_radius).reshape(
                                                          self.nrows,
                                                          self.ncols,
                                                          1)
        # if update_map:
        #     self.som_map = self.modify_weight_matrix(learning_rate,
        #                                              neighbor_influence)

        self.som_map = self.modify_weight_matrix(learning_rate,
                                                 neighbor_influence)

    def train_recurrent_som_continuous_update(self):
        for i in tqdm(range(self.num_iteration)):
            random_index = np.random.randint(low=0,
                                             high=self.input_len)
            cycle_index = []
            for j in range(self.num_cycle + 1):
                cycle_index.append(random_index - j)

            cycle_index.reverse()
            cycle_index = [x for x in cycle_index if x >= 0]
            for k in cycle_index:
                for repeat in range(self.num_repeat):
                    self.rsom_update(i, self.input_data[k])
            self.reset()

    def train_recurrent_som_sequential_data(self):
        for i in tqdm(range(self.num_iteration)):
            index = i % self.input_len
            cycle_index = []
            for j in range(self.num_cycle + 1):
                cycle_index.append(index - j)

            cycle_index.reverse()
            cycle_index = [x for x in cycle_index if x >= 0]
            for k in cycle_index:
                for repeat in range(self.num_repeat):
                    self.rsom_update(i, self.input_data[k])
            self.reset()

    def train_recurrent_som(self):
        for i in tqdm(range(self.num_iteration)):
            random_train_data_index = np.random.randint(low=1,
                                                        high=self.input_len)
            cycle_index = []
            for j in range(self.num_cycle):
                cycle_index.append(random_train_data_index - j)
            cycle_index.reverse()
            counter = 0
            for k in cycle_index:
                # print(counter)
                if (counter == self.num_cycle - 1):
                    # print("In normal cycle")
                    # print("i: {0}, k: {1}".format(i, k))
                    self.rsom_update(i, self.input_data[k], update=True)
                else:
                    # print("In final cycle")
                    # print("i: {0}, k: {1}".format(i, k))
                    self.rsom_update(i, self.input_data[k], update=False)
                counter += 1
            self.reset()

    def reset(self):
        self.differences = np.zeros(self.som_map.shape)
        # print(self.differences)
        # r_i = np.random.randint(len(self.differences))
        # r_j = np.random.randint(len(self.differences))
        # self.differences[r_i][r_j] = 0.001


def train_som(som_obj):
    for i in tqdm(range(som_obj.num_iteration)):
        som_obj.som_update(i)
        # print(i)


def get_bmus(som_obj, data, bmus=[]):
    bmus = []
    print(len(data))
    for i in range(len(data)):
        # print("i: {0}, data: {1}".format(i, data[i]))
        bmus.append(som_obj.get_bmu(data[i]))
    bmus = np.array(bmus)
    return bmus


def return_bmu_weights(som_obj, bmus, weights=[]):
    weights = []
    for i in range(len(bmus)):
        bmu = bmus[i]
        x_loc = bmu[0]
        y_loc = bmu[1]
        weights.append(som_obj.som_map[x_loc][y_loc])
    weights = np.array(weights)
    return weights


# def check_rsom(weights, data):
#     x = range(0, len(weights))
#     # x = weights[:, 0]
#     y = weights[:, 1]
#     da_x = data[:, 0]
#     da_y = data[:, 1]
#     fig, ax = plt.subplots()
#     ax.scatter(x, y, c='r')
#     ax.scatter(da_x, da_y, c='b')
#     # for i in range(len(data)):
#     #     ax.annotate(str(data[i]), (x[i], y[i]))
#     # ax.set_xlim(1, np.max(x))
#     # ax.set_ylim(1, np.max(y))
#     # ax.set_xticks(xx)
#     # ax.set_yticks(xx)
#     plt.show()


def print_bmus_weight(data, weights):
    for i in range(len(data)):
        print("data: {0}, wgs: {1}".format(data[i], weights[i]))
