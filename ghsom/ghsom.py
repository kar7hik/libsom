import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import cog_arch
from queue import Queue
from multiprocessing import Pool


class Neuron():
    def __init__(self,
                 location,
                 weight_vector,
                 growing_metric,
                 tau_2,
                 zero_quantization_error):
        """
        zero_quantization_error: The quantization error of the layer 0

        """
        self.x_location = location[0]
        self.y_location = location[1]
        self.weight_vector = weight_vector
        self.growing_metric = growing_metric

        # TODO: Better name for tau_2
        self.tau_2 = tau_2
        self.zero_quantization_error = zero_quantization_error
        self.child_map = None
        self.input_dataset = None

        self.previous_count = 0
        self.current_count = 0

    def set_weight_vector(self, weight_vector):
        self.weight_vector = weight_vector

    def set_location(self, location):
        self.x_location = location[0]
        self.y_location = location[1]

    def get_weight_vector(self):
        return self.weight_vector

    def get_location(self):
        return np.array([self.x_location, self.y_location])

    def activation(self, datapoint):
        activation = np.linalg.norm(np.subtract(datapoint, self.weight_vector),
                                    ord=2)
        return activation

    def find_distance(self, dataset):
        """
        Returns the Euclidean norm between weight vector and the whole \
        input data set
        """
        distance = np.linalg.norm(np.subtract(self.weight_vector,
                                              dataset),
                                  ord=2, axis=1)
        return distance

    def needs_child_map(self):
        """
        Returns the Boolean value.
        Used to decide whether this neuron needs more expansion.
        """
        return self.compute_quantization_error() >= (self.tau_2 *
                                                     self.zero_quantization_error)

    def compute_quantization_error(self, growing_metric=None):
        """
        Returns the mean quantization error
        """
        if growing_metric is None:
            growing_metric = self.growing_metric
        distance_from_whole_dataset = self.find_distance(self.input_dataset)
        quantization_error = distance_from_whole_dataset.sum()

        if (self.growing_metric == "mqe"):
            quantization_error /= self.input_dataset.shape[0]
            return quantization_error
        else:
            return quantization_error

    def find_weight_distance_to_other_unit(self, other_unit):
        return np.linalg.norm(self.weight_vector - other_unit.weight_vector)

    def has_dataset(self):
        return len(self.input_dataset) != 0

    def replace_dataset(self, datapoint):
        self.current_count = len(datapoint)
        self.input_dataset = datapoint
        
    def clear_dataset(self):
        self.previous_count = self.current_count
        self.current_count = 0
        self.input_dataset = self.init_empty_dataset()

    def has_changed_from_previous_epoch(self):
        return self.previous_count != self.current_count

    def init_empty_dataset(self):
        num_input_features = self.weight_vector.shape[0]
        return np.empty(shape=(0, num_input_features), dtype=np.float32)

    def __repr__(self):
        printable = "position {} -- map dimensions {} \
        -- input dataset {} -- \n".format(self.get_location(),
                                          self.weight_vector.shape,
                                          self.input_dataset.shape[0])
        if self.child_map is not None:
            for neuron in self.child_map.neurons.values():
                printable += neuron.__repr__()

        return printable


class Neuron_Creator():
    zero_unit_quantization_error = None
    def __init__(self, tau_2, growing_metric):
        # TODO: Better name for tau_2
        self.tau_2 = tau_2
        self.growing_metric = growing_metric

    def new_neuron(self, neuron_location, weight_vector):
        assert self.zero_unit_quantization_error is not None, \
            "Unit zero's quantization error has not been set yet."
        return Neuron(neuron_location,
                      weight_vector,
                      self.growing_metric,
                      self.tau_2,
                      self.zero_unit_quantization_error)

    def zero_neuron(self, input_dataset):
        zero_neuron = Neuron((0, 0),
                             self._calc_input_mean(input_dataset),
                             self.growing_metric,
                             None,
                             None)
        zero_neuron.input_dataset = input_dataset
        self.zero_unit_quantization_error = zero_neuron.compute_quantization_error()
        
        return zero_neuron

    @staticmethod
    def _calc_input_mean(input_dataset):
        return input_dataset.mean(axis=0)


class Growing_SOM():
    def __init__(self,
                 initial_map_size,
                 tau_1,
                 weight_map,
                 parent_dataset,
                 parent_quantization_error,
                 neuron_creator,
                 # init_learning_rate,
                 # init_radius,
                 # time_constant,
                 native_update=True):
        """Constructor for Growing_SOM Class."""
        self.initial_map_size = initial_map_size
        self.weight_map = weight_map
        self.parent_dataset = parent_dataset
        self.parent_quantization_error = parent_quantization_error
        self.num_input_features = parent_dataset.shape[1]
        self.tau_1 = tau_1
        self.neuron_creator = neuron_creator
        self.neurons = self.create_neurons()
        self.current_som_map = weight_map
        self.som = None
        self.current_map_shape = self.initial_map_size

        # self.node_list = np.array(list(
        #     itertool.product(range(self.current_map_shape[0]),
        #                      range(self.current_map_shape[1]))),
        #                           dtype=int)
        # self.init_learning_rate = init_learning_rate
        # self.init_radius = init_radius
        # self.time_constant = time_constant
        
    def create_neurons(self):
        rows, cols = self.initial_map_size
        neurons = {(x, y): self.create_neuron((x, y))
                                              for x in range(rows)
                                              for y in range(cols)}
        return neurons

    def create_neuron(self, location):
        neuron = self.neuron_creator.new_neuron(location,
                                                self.weight_map[location[0],
                                                                location[1]])
        return neuron

    # def decay_radius(self, curr_iter):
    #     return self.init_radius * np.exp(-current_iter / self.time_constant)

    # def decay_learning_rate(self, curr_iter):
    #     return self.init_learning_rate * \
    #         np.exp(-current_iter / self.num_iteration)

    def train(self, epochs, num_iteration, learning_rate):
        iteration = 0
        can_grow = True

        while can_grow and (iteration < num_iteration):
            self.neurons_training(epochs,
                                  learning_rate,
                                  self.parent_dataset)
            iteration += 1
            can_grow = self.allowed_to_grow()
            if can_grow:
                # print("Allowed to grow")
                # print("Current iteration: {}".format(iteration))
                self.grow()
                # print(self.weight_map.shape)
            else:
                # print("Cannot grow")
                pass

        if can_grow:
            self.map_data_to_neurons()

        return self

    # def training_neuron(self, epochs, learning_rate, decay=0.95, sigma=1.5):
    #     lr = learning_rate
    #     s = sigma

    #     for iteration in range(epochs):
    #         for data in self.__training_data(seed,
    #                                          dataset_percentage,
    #                                          min_dataset_size):
    #             self.__update_neurons(data, lr, s)

    #         lr *= decay
    #         s *= decay


    def neurons_training(self,
                         epochs,
                         learning_rate,
                         input_dataset):
        nrows, ncols = self.current_map_shape
        self.som = cog_arch.SOM(nrows,
                                ncols,
                                input_dataset,
                                num_iteration=epochs,
                                init_learning_rate=learning_rate,
                                init_radius=1.5)
        dataset_percentage = 0.50
        min_size = 30
        seed = None
        neighbor_radius = 1.5

        # print("Map shape: {}, weight shape: {}".format(self.current_map_shape,
        #                                                self.weight_map.shape))
        self.som.som_map = self.weight_map
        self.som.batch_train_som(dataset_percentage,
                                 min_size,
                                 seed)
        self.current_som_map = self.weight_map

    def allowed_to_grow(self):
        self.map_data_to_neurons()
        
        MQE = 0.0
        mapped_neurons = 0
        changed_neurons = 0

        assert self.parent_quantization_error is not None, \
            "Parent Quantization Error must not be None"

        for neuron in self.neurons.values():
            changed_neurons += 1 if neuron.has_changed_from_previous_epoch() else 0
            if neuron.has_dataset():
                MQE =+ neuron.compute_quantization_error()
                mapped_neurons += 1

        if mapped_neurons == 0:
            pass
            # print("Current neuron doesn't have any dataset assigned to it")
        #     print("Som map shape: {}".format(self.current_map_shape))
        #     print("Som map: {}".format(self.current_som_map))
        #     print("Newline")
        #     print("Parent data shape: {}".format(self.parent_dataset.shape))
        #     print("Parent data: {}".format(self.parent_dataset))
        #     for idx, neuron in enumerate(self.neurons.values()):
        #         print("neuron {} data set shape: {}".format(idx, neuron.input_dataset.shape))

        # print("Mapped neurons: {}, changed_neurons: {}".format(mapped_neurons,
        #                                                        changed_neurons))
        return ((MQE / mapped_neurons) >= (self.tau_1 *
                                           self.parent_quantization_error)) and \
                                           (changed_neurons > int(np.round(mapped_neurons / 5)))

    def map_data_to_neurons(self):
        input_dataset = self.parent_dataset
        for neuron in self.neurons.values():
            neuron.clear_dataset()

        neurons = list(self.neurons.values())
        num_neurons = len(neurons)
        map_shape = (self.weight_map.shape[0], self.weight_map.shape[1])
        # print("Neurons shape: {}, num neurons: {}".format(neurons_shape,
        #                                                   num_neurons))
        bmus = []
        for i in input_dataset:
            bmu = self.som.get_bmu(i)
            bmus.append(bmu)
        bmus = np.asarray(bmus)
        
        bmu_as_index = []
        for bmu in bmus:
            bmu_as_index.append(np.ravel_multi_index(bmu, map_shape))

        neurons_as_index = []
        for neuron in self.neurons.values():
            neuron_location = neuron.get_location()
            neurons_as_index.append(np.ravel_multi_index(neuron_location,
                                                         map_shape))
        
        data_map_to_neuron = [[location for location in np.where(bmu_as_index == neuron_idx)[0]]
                              for neuron_idx in neurons_as_index]

        # if len(data_map_to_neuron) == 0:
        #     print("Bmus: {}".format(bmus))
        #     print("bmu as index: {}".format(bmu_as_index))
        #     print("neurons as index: {}".format(neurons_as_index))

        # print("Bmus: {}".format(bmus))
        # print("bmu as index: {}".format(bmu_as_index))
        # print("neurons as index: {}".format(neurons_as_index))
        # print("Data map to neuron: {}".format(np.array(data_map_to_neuron)))
 
        for idx, data_idxs in enumerate(data_map_to_neuron):
            
            neurons[idx].replace_dataset(input_dataset[data_idxs, :])
            # print("mapped dataset shape: {}".format(input_dataset[data_idxs, :].shape))
            # print("Parent dataset shape: {}".format(input_dataset.shape))

    def find_error_neuron(self):
        quantization_errors = list()
        for neuron in self.neurons.values():
            quantization_error = -np.inf
            if neuron.has_dataset():
                quantization_error = neuron.compute_quantization_error()
            quantization_errors.append(quantization_error)

        # print("Current map shape: {}".format(self.current_map_shape))
        error_neuron_index = np.unravel_index(np.argmax(quantization_errors),
                                              dims=self.current_map_shape)
        error_list_index = np.argmax(quantization_errors)
        # print("error index: {}".format(error_neuron_index))
        error_neuron = self.neurons[error_neuron_index]
        return error_neuron, error_neuron_index

    def find_most_dissimilar_neuron(self, error_neuron):
        weight_distances = dict()

        for neuron in self.neurons.values():
            if self.are_neighbors(error_neuron, neuron):
                weight_distance = error_neuron.find_weight_distance_to_other_unit(neuron)
                weight_distances[neuron] = weight_distance

        dissimilar_neuron = max(weight_distances, key=weight_distances.get)
        return dissimilar_neuron

    def grow(self):
        error_neuron, error_neuron_index = self.find_error_neuron()
        dissimilar_neuron = self.find_most_dissimilar_neuron(error_neuron)

        if self.are_in_same_column(error_neuron, dissimilar_neuron):
            # print("Same column")
            new_column_indices = self.add_column_in_between(error_neuron,
                                                            dissimilar_neuron)
            self.init_new_weight_vector(new_column_indices, "horizontal")

        elif self.are_in_same_row(error_neuron, dissimilar_neuron):
            # print("Same row")
            new_row_indices = self.add_row_in_between(error_neuron,
                                                      dissimilar_neuron)
            self.init_new_weight_vector(new_row_indices, "vertical")


    def add_column_in_between(self, error_neuron, dissimilar_neuron):
        """
        Adds an entire column between the error neuron and the dissimilar
        neuron
        """
        error_col = error_neuron.get_location()[1]
        dissimilar_col = dissimilar_neuron.get_location()[1]
        new_column_idx = max(error_col, dissimilar_col)
        map_rows, map_cols = self.current_map_shape

        # new_column_idx = 2 then [(0, 2), (1, 2), (2, 2)]
        new_line_idx = [(row, new_column_idx) for row in range(map_rows)]
        # print("new_col_line_idx: {}".format(new_line_idx))        


        for row in range(map_rows):
            for col in reversed(range(new_column_idx, map_cols)):
                new_idx = (row, col+1)
                neuron = self.neurons.pop((row, col))
                neuron.set_location(new_idx)
                self.neurons[new_idx] = neuron

        line = np.zeros(shape=(map_rows, self.num_input_features),
                        dtype=np.float32)
        self.weight_map = np.insert(self.weight_map,
                                    new_column_idx,
                                    line,
                                    axis=1)

        # Update the current map shape
        # self.current_map_shape = self.neurons.shape
        self.current_map_shape = (self.weight_map.shape[0], self.weight_map.shape[1])
        # print("Current neurons map shape: {}".format(self.current_map_shape))
        return new_line_idx
            

    def print_neurons_matrix(self):
        # print("Current neurons map shape: {}".format(self.neurons.shape))
        for idx, neuron in enumerate(self.neurons.values()):
            neuron_location = neuron.get_location()
            # print("Neuron {} location: {}".format(idx, neuron_location))

    def add_row_in_between(self, error_neuron, dissimilar_neuron):
        """
        Adds an entire row between the error neuron and the dissimilar
        neuron
        """
        error_row = error_neuron.get_location()[0]
        dissimilar_row = dissimilar_neuron.get_location()[0]
        new_row_idx = max(error_row, dissimilar_row)
        map_rows, map_cols = self.current_map_shape

        
        # new_row_idx = 2 then [(2, 0), (2, 1), (2, 2)]
        new_line_idx = [(new_row_idx, col) for col in range(map_cols)]
        # print("new_row_line_idx: {}".format(new_line_idx))

        for row in reversed(range(new_row_idx, map_rows)):
            for col in range(map_cols):
                new_idx = (row+1, col)
                neuron = self.neurons.pop((row, col))
                neuron.set_location(new_idx)
                self.neurons[new_idx] = neuron

        line = np.zeros(shape=(map_cols, self.num_input_features),
                        dtype=np.float32)
        self.weight_map = np.insert(self.weight_map,
                                    new_row_idx,
                                    line,
                                    axis=0)
                
        # Update the current map shape
        self.current_map_shape = (self.weight_map.shape[0], self.weight_map.shape[1])
        return new_line_idx

    def init_new_weight_vector(self, new_neuron_idxs, direction):
        for row, col in new_neuron_idxs:
            # empty_weight_vector = np.zeros(shape=(self.num_input_features))
            # self.neurons[(row, col)] = self.create_neuron([row, col],
            #                                               empty_weight_vector)
            adjacent_neuron_idxs = self.get_adjacent_neuron_idxs(row,
                                                                 col,
                                                                 direction)
            weight_vector = self.mean_weight_vector(adjacent_neuron_idxs)
            # print("row: {0}, col: {1}".format(row, col))
            # print(weight_vector)
            self.weight_map[row, col] = weight_vector
            self.neurons[(row, col)] = self.create_neuron((row, col))
            
    @staticmethod
    def get_adjacent_neuron_idxs(row, col, direction):
        adjacent_neuron_idxs = []
        if direction == "horizontal":
            adjacent_neuron_idxs = [(row, col - 1), (row, col + 1)]

        elif direction == "vertical":
            adjacent_neuron_idxs = [(row - 1, col), (row + 1, col)]

        return adjacent_neuron_idxs

    def mean_weight_vector(self, neuron_idxs):
        weight_vector = np.zeros(shape=self.num_input_features, dtype=np.float32)
        for adjacent_idx in neuron_idxs:
            # print(adjacent_idx)
            weight_vector += 0.5 * self.neurons[adjacent_idx].get_weight_vector()
        return weight_vector

    @staticmethod
    def are_neighbors(first_neuron, second_neuron):
        return np.linalg.norm(np.asarray(first_neuron.get_location()) -
                              np.asarray(second_neuron.get_location()),
                              ord=1) == 1

    @staticmethod
    def are_in_same_column(first_neuron, second_neuron):
        """
        Checks whether the two neurons are in the same column.
        If they are in same column, the difference would be equal to zero.
        """
        first_neuron_col_value = first_neuron.get_location()[1]
        second_neuron_col_value = second_neuron.get_location()[1]        

        return abs(first_neuron_col_value - second_neuron_col_value) == 0

    @staticmethod
    def are_in_same_row(first_neuron, second_neuron):
        """
        Checks whether the two neurons are in the same row.
        If they are in same row, the difference would be equal to zero.
        """
        first_neuron_row_value = first_neuron.get_location()[0]
        second_neuron_row_value = second_neuron.get_location()[0]        

        return abs(first_neuron_row_value - second_neuron_row_value) == 0

    def plot_data(self):
        rows = self.current_som_map.shape[0]
        cols = self.current_som_map.shape[1]
        som_map = self.current_som_map
        
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

    def map_shape(self):
        shape = self.weight_map.shape
        return shape[0], shape[1]

class Create_Neuron_Layer():
    def __init__(self,
                 tau_1,
                 parent_quantization_error,
                 parent_dataset,
                 neuron_creator):
        """Constructor for Neuron layer Creator"""
        self.tau_1 = tau_1
        self.parent_quantization_error = parent_quantization_error
        self.parent_dataset = parent_dataset
        self.weight_map = weight_map
        self.neuron_creator = neuron_creator


        self.zero_unit = self.create_zero_neuron()

    def create_zero_neuron(self):
        zero_neuron = self.neuron_creator.zero_neuron(self.input_dataset)
        zero_neuron.child_map = self.create_new_GSOM(zero_neuron.parent_quantization_error,
                                                     zero_neuron.parent_dataset,
                                                     self.weight_map)

    def create_new_GSOM(self,
                        parent_quantization_error,
                        parent_dataset,
                        weight_map):
        return Growing_SOM((2, 2),
                           self.tau_1,
                           weight_map,
                           parent_dataset,
                           parent_quantization_error,
                           neuron_creator)



class GHSOM():
    def __init__(self,
                 input_dataset,
                 tau_1,
                 tau_2,
                 learning_rate,
                 growing_metric="qe",
                 native_update=True):
        """Growing Hierarchical Self Organizing Map Class"""
        self.input_dataset = input_dataset
        self.num_input_features = self.input_dataset.shape[1]
        self.learning_rate = learning_rate
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.growing_metric = growing_metric
        self.neuron_creator = Neuron_Creator(self.tau_2, self.growing_metric)
        self.native_update =native_update

    def train(self, epochs=20):
        zero_unit = self.create_zero_neuron()

        neurons_to_train_queue = Queue()
        neurons_to_train_queue.put(zero_unit)

        pool = Pool(processes=None)
        while not neurons_to_train_queue.empty():
            size = min(neurons_to_train_queue.qsize(), pool._processes)
            growing_maps = dict()
            for i in range(size):
                neuron = neurons_to_train_queue.get()
                growing_maps[neuron] = (pool.apply_async(neuron.child_map.train,
                                                         (epochs,
                                                          10,
                                                          self.learning_rate)))

            for neuron in growing_maps:
                growing_map = growing_maps[neuron].get()
                neuron.child_map = growing_map
                neurons_to_expand = filter(lambda _neuron: _neuron.needs_child_map(),
                                           growing_map.neurons.values())

                for _neuron in neurons_to_expand:
                    # print("weight map of growing_map:")
                    # print(growing_map.weight_map)
                    _neuron.child_map = self.create_new_GSOM(
                        _neuron.compute_quantization_error(),
                        _neuron.input_dataset,
                        self.assign_weights_to_new_gsom(_neuron.get_location(),
                                                        growing_map.weight_map))
                    neurons_to_train_queue.put(_neuron)
                
        return zero_unit

    def create_zero_neuron(self):
        zero_neuron =  self.neuron_creator.zero_neuron(self.input_dataset)
        zero_neuron.child_map = self.create_new_GSOM(
            self.neuron_creator.zero_unit_quantization_error,
            zero_neuron.input_dataset,
            self.calc_initial_random_weights())
        return zero_neuron

    def create_new_GSOM(self,
                        parent_quantization_error,
                        parent_dataset,
                        weight_map):
        return Growing_SOM((2, 2),
                           self.tau_1,
                           weight_map,
                           parent_dataset,
                           parent_quantization_error,
                           self.neuron_creator)

    def calc_initial_random_weights(self):
        random_weights = np.zeros(shape=(2, 2, self.num_input_features))
        
        for location in np.ndindex(2, 2):
            random_data_index = np.random.randint(len(self.input_dataset))
            random_data_item = self.input_dataset[random_data_index]
            random_weights[location] = random_data_item

        return random_weights

    def assign_weights_to_new_gsom(self, parent_position, weight_map):
        child_weights = np.zeros(shape=(2, 2, self.num_input_features))
        stencil = self.generate_kernal_stencil(parent_position)
        for child_position in np.ndindex(2, 2):
            child_position = np.asarray(child_position)
            # print(weight_map.shape)
            mask = self.filter_out_of_bound_position(child_position,
                                                     stencil,
                                                     [weight_map.shape[0], weight_map.shape[1]])
            weight = np.mean(self.element_from_position_list(weight_map,
                                                             mask),
                             axis=0)
            # print(weight)
            child_weights[child_position] = weight

        return child_weights

    def filter_out_of_bound_position(self,
                                     child_position,
                                     stencil,
                                     map_shape):
        return np.asarray(list(filter(lambda pos: self.check_position(pos,
                                                                      map_shape),
                                      stencil + child_position)))

    @staticmethod
    def element_from_position_list(matrix, positions_list):
        return matrix[positions_list[:, 0], positions_list[:, 1]]

    @staticmethod
    def generate_kernal_stencil(parent_position):
        row, col = parent_position
        return np.asarray([
            (r, c)
            for r in range(row - 1, row + 1)
            for c in range(col - 1, col + 1)
        ])

    def check_position(self, position, map_shape):
        row, col = position
        map_rows, map_cols = map_shape
        return (row >= 0 and col >= 0) and (row < map_rows and col < map_cols)


class Zero_layer():
    def __init__(self, input_dataset, neuron_creator):
        self.input_dataset = input_dataset
        self.neuron_creator = neuron_creator
        self.zero_unit = self.create_zero_unit()

    def create_zero_unit(self):
        zero_unit = self.neuron_creator.zero_neuron(self.input_dataset)
        return zero_unit
