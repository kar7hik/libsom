import numpy as np
from matplotlib import pyplot as plt
from queue import Queue
from multiprocessing import Pool
from matplotlib import patches as patches
import itertools
from math import ceil
from tqdm import tqdm
from timeit import default_timer as timer


class Neuron():
    def __init__(self,
                 location,
                 weight_vector,
                 growing_metric,
                 hierarchical_growing_coefficient,
                 zero_quantization_error):
        self.x_location, self.y_location = location
        self.weight_vector = weight_vector
        self.growing_metric = growing_metric

        self.hierarchical_growing_coefficient = hierarchical_growing_coefficient
        self.zero_quantization_error = zero_quantization_error
        self.child_map = None

        self.input_dataset = None
        self.num_input_features = self.weight_vector.shape[0]

        self.previous_count = 0
        self.current_count = 0

    def set_weight_vector(self, weight_vector):
        """Sets the weight vector of the neuron"""
        self.weight_vector = weight_vector

    def set_location(self, location):
        """Sets the location of the neuron"""
        self.x_location = location[0]
        self.y_location = location[1]

    def get_weight_vector(self):
        """Returns the weight vector of the current neuron"""
        return self.weight_vector

    def get_location(self):
        """Returns the location of the current neuron"""
        return (self.x_location, self.y_location)

    def activation(self, data):
        """
        Returns the Euclidean norm between neuron's weight vector and the \
        input data.
        """
        activation = 0

        if (len(data.shape) == 1):
            activation = np.linalg.norm(np.subtract(data,
                                                    self.weight_vector), ord=2)
        else:
            activation = np.linalg.norm(np.subtract(data,
                                                    self.weight_vector), ord=2,
                                        axis=1)
        return activation

    def compute_quantization_error(self, growing_metric=None):
        """
        Returns the Quantization error
        """
        if growing_metric is None:
            growing_metric = self.growing_metric
        distance_from_whole_dataset = self.activation(self.input_dataset)
        quantization_error = distance_from_whole_dataset.sum()

        if (self.growing_metric == "mqe"):
            quantization_error /= self.input_dataset.shape[0]

        return quantization_error

    def needs_child_map(self):
        """
        Returns the Boolean value.
        Used to decide whether this neuron needs more expansion.
        """
        return self.compute_quantization_error() >= (self.hierarchical_growing_coefficient *
                                                     self.zero_quantization_error)

    def find_weight_distance_to_other_unit(self, other_unit):
        """
        Returns the distance from the current neuron to the other neuron
        """

        return np.linalg.norm(self.weight_vector - other_unit.weight_vector)

    def has_dataset(self):
        """
        Returns the Boolean value. Checks whether the current neuron has \
        any data set
        """
        return len(self.input_dataset) != 0

    def init_empty_dataset(self):
        """
        Resets the current input dataset to an empty dataset with the \
        dimension of 0 and the number of input data features.
        """
        return np.empty(shape=(0,
                               self.num_input_features),
                        dtype=np.float32)

    def replace_dataset(self, dataset):
        """
        Replaces the current input dataset of the neuron with the given dataset
        """
        self.current_count = len(dataset)
        self.input_dataset = dataset

    def clear_dataset(self):
        """
        Clears the current input dataset of the neuron
        """
        self.previous_count = self.current_count
        self.current_count = 0
        self.input_dataset = self.init_empty_dataset()

    def has_changed_from_previous_epoch(self):
        return self.previous_count != self.current_count


class Neuron_Creator():
    zero_quantization_error = None

    def __init__(self,
                 hierarchical_growing_coefficient,
                 simple=False,
                 growing_metric="qe"):
        self.hierarchical_growing_coefficient = hierarchical_growing_coefficient
        self.growing_metric = growing_metric

    def new_neuron(self, neuron_location, weight_vector):
        assert self.zero_quantization_error is not None, \
            "Unit zero's quantization error has not been set yet."
        return Neuron(neuron_location,
                      weight_vector,
                      self.growing_metric,
                      self.hierarchical_growing_coefficient,
                      self.zero_quantization_error)

    def zero_neuron(self, input_dataset):
        zero_neuron = Neuron((0, 0),
                             self._calc_input_mean(input_dataset),
                             self.growing_metric,
                             None,
                             None)
        zero_neuron.input_dataset = input_dataset
        self.zero_quantization_error = zero_neuron.compute_quantization_error()

        return zero_neuron

    @staticmethod
    def _calc_input_mean(input_dataset):
        return input_dataset.mean(axis=0)


class SOM():
    def __init__(self,
                 nrows,
                 ncols,
                 input_data,
                 initial_learning_rate=0.9,
                 initial_neighbor_radius=1.5,
                 epochs=15,
                 initialization="random"):
        self.nrows = nrows
        self.ncols = ncols
        self.input_data = input_data
        self.initial_learning_rate = initial_learning_rate
        self.initial_neighbor_radius = initial_neighbor_radius
        self.epochs = epochs
        self.time_constant = self.epochs / np.log(self.initial_neighbor_radius)
        self.initialization = initialization


        # Input data parameters
        self.input_data_length = self.input_data.shape[0]
        self.num_input_features = self.input_data.shape[1]
        

        if (self.initialization == "random"):
            self.som_map = np.random.random((self.nrows,
                                             self.ncols,
                                             self.num_input_features))
        else:
            self.som_map = None

        self.neuron_list = np.array(list(
            itertools.product(range(self.nrows),
                              range(self.ncols))),
                                    dtype=int)

    def create_batch_data(self,
                          dataset_percentage,
                          min_dataset_size,
                          seed):
        dataset_size = self.input_data_length
        if dataset_size <= min_dataset_size:
            iterator = range(dataset_size)
        else:
            iterator = range(int(ceil(dataset_size * dataset_percentage)))

        # print("Selected size: {}".format(iterator))
        random_generator = np.random.RandomState(seed)
        for dummy in iterator:
            yield random_generator.randint(dataset_size)

    def decay_neighbor_radius(self, current_iter):
        return self.initial_neighbor_radius * \
            np.exp(-current_iter / self.time_constant)

    def decay_learning_rate(self, current_iter):
        return self.initial_learning_rate * \
            np.exp(-current_iter / self.epochs)

    def calculate_influence(self, w_dist, neighbor_radius):
        pseudogaussian = np.exp(-np.divide(np.power(w_dist, 2),
                                           (2 * np.power(neighbor_radius, 2))))
        return pseudogaussian

    def calculate_influence_v2(self, w_dist, neighbor_radius):
        return np.exp(-w_dist / (2 * (neighbor_radius ** 2)))


    def get_bmu(self, datapoint):
        # print(self.som_map)
        a = np.linalg.norm(self.som_map - datapoint, axis=2)
        # print(a)
        bmu_index = np.argwhere(a == np.min(a))[0]
        # print("BMU Index: {}".format(bmu_index))
        # bmu = self.som_map[bmu_index]
        return tuple(bmu_index)

    def som_update(self, datapoint, learning_rate, neighbor_radius):
        winner_neuron_index = self.get_bmu(datapoint)
        w_dist = np.linalg.norm(self.neuron_list - winner_neuron_index, axis=1)
        neighbor_influence = self.calculate_influence(w_dist,
                                                      neighbor_radius).reshape(
                                                          self.nrows,
                                                          self.ncols,
                                                          1)
        self.som_map = self.modify_weight_matrix(learning_rate,
                                                 neighbor_influence,
                                                 datapoint)

    def som_train(self):
        """
        Interface to train the SOM.
        """
        lr = self.initial_learning_rate
        nr = self.initial_neighbor_radius
        epochs = self.epochs

        for epoch in tqdm(range(epochs)):
            random_index = np.random.randint(low=0,
                                             high=self.input_data_length)
            datapoint = self.input_data[random_index]
            self.som_update(datapoint, lr, nr)
            # print("learning_rate: {}, neighbor_radius: {}".format(lr, nr))

            lr, nr = self.find_learning_parameters(epoch,
                                                   datapoint,
                                                   lr,
                                                   nr)

    def som_batch_train(self,
                        dataset_percentage,
                        min_dataset_size):
        lr = self.initial_learning_rate
        nr = self.initial_neighbor_radius
        epochs = self.epochs

        for i in tqdm(range(epochs)):
            for data_index in self.create_batch_data(dataset_percentage,
                                                     min_dataset_size,
                                                     seed=None):
                data = self.input_data[data_index]
                self.som_update(data, lr, nr)
                lr, nr = self.find_learning_parameters(i, data, lr, nr)

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

    def find_learning_parameters(self,
                                 current_iter,
                                 datapoint,
                                 learning_rate,
                                 neighbor_radius):
        """
        Performs the computation of learning rate and neighbor radius
        """
        learning_rate = self.decay_learning_rate(current_iter)
        neighbor_radius = self.decay_neighbor_radius(current_iter)

        return learning_rate, neighbor_radius


class RSOM(SOM):
    def __init__(self,
                 nrows,
                 ncols,
                 input_data,
                 initial_learning_rate=0.25,
                 initial_neighbor_radius=1.5,
                 epochs=15,
                 initialization="random"):
        SOM.__init__(self,
                     nrows,
                     ncols,
                     input_data,
                     initial_learning_rate,
                     initial_neighbor_radius,
                     epochs,
                     initialization=initialization)

        self.num_cycle = 0
        self.num_repeat = 0
        self.alpha = 0
        self.differences = None

    def find_difference_vector(self, datapoint, alpha):
        """
        Parameters:
        y_t: Y at time t
        x_t: X at time t
        w_t: weight value at t
        alpha: Weighting factor determining the effect of
            the difference vector
        """
        # print(self.differences)
        first = ((1 - alpha) * self.differences)
        second = (alpha * (datapoint - self.som_map))
        y_t = first + second
        # print(y_t)
        return y_t

    def get_rsom_bmu(self, datapoint, alpha):
        self.differences = self.find_difference_vector(datapoint, alpha)
        a = np.linalg.norm(self.differences, axis=2)
        return np.argwhere(a == np.min(a))[0]

    def reset(self):
        self.differences = np.zeros(self.som_map.shape)

    def rsom_update(self,
                    datapoint,
                    learning_rate,
                    neighbor_radius):
        winner_neuron_index = self.get_rsom_bmu(datapoint, self.alpha)
        w_dist = np.linalg.norm(self.neuron_list - winner_neuron_index, axis=1)
        neighbor_influence = self.calculate_influence(w_dist,
                                                      neighbor_radius).reshape(
                                                          self.nrows,
                                                          self.ncols,
                                                          1)
        self.som_map = self.modify_weight_matrix(learning_rate,
                                                 neighbor_influence,
                                                 datapoint)

    def rsom_batch_train(self,
                         num_iteration=100,
                         num_cycle=5,
                         num_repeat=2,
                         alpha=0.7,
                         dataset_percentage=0.5,
                         min_dataset_size=10):
        self.num_cycle = num_cycle
        self.num_repeat = num_repeat
        self.alpha = alpha
        self.epochs = num_iteration

        lr = self.initial_learning_rate
        nr = self.initial_neighbor_radius
        epochs = self.epochs

        assert (self.som_map is not None), \
            "Please initialize SOM weight map!!!"
        self.differences = np.zeros(self.som_map.shape)

        for iteration in tqdm(range(epochs)):
            for data_index in self.create_batch_data(dataset_percentage,
                                                     min_dataset_size,
                                                     seed=None):
                cycle_index = []
                for j in range(self.num_cycle + 1):
                    cycle_index.append(data_index - j)

                    cycle_index.reverse()
                    cycle_index = [index for index in cycle_index if index >= 0]

                for k in cycle_index:
                    for repeat in range(self.num_repeat):
                        self.rsom_update(self.input_data[k],
                                         lr,
                                         nr)

                self.reset()
                lr, nr = self.find_learning_parameters(iteration,
                                                       self.input_data[data_index],
                                                       lr,
                                                       nr)

    def rsom_train(self,                      
                   num_iteration=100,
                   num_cycle=5,
                   num_repeat=2,
                   alpha=0.7):
        self.num_cycle = num_cycle
        self.num_repeat = num_repeat
        self.alpha = alpha
        self.num_iteration = num_iteration

        assert (self.som_map is not None), \
            "Please initialize SOM weight map!!!"
        self.differences = np.zeros(self.som_map.shape)

        lr = self.initial_learning_rate
        nr = self.initial_neighbor_radius

        for iteration in tqdm(range(self.num_iteration)):
            random_index = np.random.randint(low=0,
                                             high=self.input_data_length)
            cycle_index = []
            for j in range(self.num_cycle + 1):
                cycle_index.append(random_index - j)

            cycle_index.reverse()
            cycle_index = [index for index in cycle_index if index >= 0]

            for k in cycle_index:
                for repeat in range(self.num_repeat):
                    self.rsom_update(self.input_data[k],
                                     lr,
                                     nr)

            self.reset()
            # print("learning_rate: {}, neighbor_radius: {}".format(lr, nr))
            lr, nr = self.find_learning_parameters(iteration,
                                                   self.input_data[random_index],
                                                   lr,
                                                   nr)

# input_dataset = np.random.random((100, 3))
class PLSOM(SOM):
    def __init__(self,
                 nrows,
                 ncols,
                 input_data,
                 initial_learning_rate=0.9,
                 initial_neighbor_radius=1.5,
                 epochs=15,
                 beta=None,
                 initialization="random"):
        SOM.__init__(self,
                     nrows,
                     ncols,
                     input_data,
                     initial_learning_rate,
                     initial_neighbor_radius,
                     epochs,
                     initialization=initialization)
        self.scaling_variable = None
                
        # Constant initial value for beta.
        self.beta = beta if beta else 2
        self.theta = 1

        self.best_fit_learning_rate = None
        self.best_fit_neighbor_radius = None
        self.neighbor_influence = None

        self.curr_bmu_index = None
        self.curr_bmu_weight = None

        self.winner_euclidean_distance = None

        self.counter = 0

    def find_winner_euclidean_distance(self, datapoint):
        self.curr_bmu_index = self.get_bmu(datapoint)
        self.curr_bmu_weight = self.som_map[self.curr_bmu_index]
        return np.linalg.norm(datapoint - self.curr_bmu_weight)
        
        
    def initialize_scaling_variable(self, datapoint):
        return self.find_winner_euclidean_distance(datapoint)

    def find_scaling_variable(self, winner_euclidean_distance):
        a = max(winner_euclidean_distance, self.scaling_variable)
        # print(winner_euclidean_distance, self.scaling_variable)
        return a

    def find_best_fit_learning_rate(self, datapoint):
        self.winner_euclidean_distance = self.find_winner_euclidean_distance(datapoint)
        self.scaling_variable = self.find_scaling_variable(self.winner_euclidean_distance)
        # if (winner_euclidean_distance == 0.0):
        #     return 0.001
        # print(self.winner_euclidean_distance, self.scaling_variable)
        return self.winner_euclidean_distance / self.scaling_variable

    def find_best_fit_neighbor_radius(self, datapoint):
        self.best_fit_neighbor_radius = (self.beta - self.theta) * \
            np.log(1 + self.best_fit_learning_rate * (np.e - 1)) + self.theta

        return self.best_fit_neighbor_radius    

    def calculate_influence(self):
        w_dist = np.linalg.norm(self.neuron_list - self.curr_bmu_index,
                                axis=1)

        # print(self.best_fit_learning_rate, self.best_fit_neighbor_radius)
        return np.exp(-np.divide((np.power(w_dist, 2)),
                                 (np.power(self.best_fit_neighbor_radius,
                                          2)))).reshape(self.nrows,
                                                        self.ncols,
                                                        1)

    def find_learning_parameters(self,
                                 epoch,
                                 datapoint,
                                 learning_rate,
                                 neighbor_radius):
        if (epoch == 0):
            self.scaling_variable = self.initialize_scaling_variable(datapoint)

        self.best_fit_learning_rate = self.find_best_fit_learning_rate(datapoint)
        self.best_fit_neighbor_radius = self.find_best_fit_neighbor_radius(datapoint)

    def plsom_update(self, datapoint):
        self.neighbor_influence = self.calculate_influence()
        self.som_map = self.modify_weight_matrix(self.best_fit_learning_rate,
                                                 self.neighbor_influence,
                                                 datapoint)

    def plsom_train(self):
        epochs = self.epochs
        for epoch in tqdm(range(epochs)):
            random_index = np.random.randint(low=0,
                                             high=self.input_data_length)
            datapoint = self.input_data[random_index]
            self.find_learning_parameters(epoch,
                                          datapoint,
                                          self.best_fit_learning_rate,
                                          self.best_fit_neighbor_radius)
            self.plsom_update(datapoint)

    def plsom_batch_train(self,
                          dataset_percentage,
                          min_dataset_size):
        epochs = self.epochs
        for epoch in tqdm(range(epochs)):
            # print("epoch: {}".format(epoch))
            for data_index in self.create_batch_data(dataset_percentage,
                                                     min_dataset_size,
                                                     seed=None):
                # print("epoch: {}".format(epoch))
                data = self.input_data[data_index]
                self.find_learning_parameters(self.counter,
                                              data,
                                              self.best_fit_learning_rate,
                                              self.best_fit_neighbor_radius)
                self.plsom_update(data)
                self.counter += 1


class PLRSOM(RSOM):
    def __init__(self):
        pass



class Growing_SOM(SOM):
    def __init__(self,
                 initial_map_size,
                 map_growing_coefficient,
                 weight_map,
                 parent_dataset,
                 parent_quantization_error,
                 neuron_creator):
        """Constructor for Growing_SOM Class."""
        self.initial_map_size = initial_map_size
        self.weight_map = weight_map
        self.parent_dataset = parent_dataset
        self.parent_quantization_error = parent_quantization_error

        # Input data parameters
        self.input_data_length = self.parent_dataset.shape[0]
        self.num_input_features = self.parent_dataset.shape[1]


        self.map_growing_coefficient = map_growing_coefficient
        self.neuron_creator = neuron_creator
        self.neurons = self.create_neurons()

        self.current_map_shape = self.initial_map_size
        self.nrows, self.ncols = self.initial_map_size

        self.initial_learning_rate = None
        self.initial_neighbor_radius = None
        self.time_constant = None
        self.epochs = None
        self.max_iter = None

        self.som = None

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

    def mapping_data_helper(self, data):
        number_of_data = None
        distances = None

        if (len(data.shape) == 1):
            number_of_data = 1
        else:
            number_of_data = data.shape[0]

        distances = np.empty(shape=(number_of_data,
                                    len(self.neurons.values())))

        neurons_list = list(self.neurons.values())
        for idx, neuron in enumerate(neurons_list):
            distances[:, idx] = neuron.activation(data)

        # winner neurons for the given input data
        winner_neuron_per_data = distances.argmin(axis=1)

        map_data_to_neurons = [[position for position in np.where(winner_neuron_per_data == neuron_idx)[0]]
                               for neuron_idx in range(len(neurons_list))]
        winner_neurons = [neurons_list[idx] for idx in winner_neuron_per_data]

        return winner_neurons, map_data_to_neurons
    
    def gsom_train(self,
                   epochs,
                   initial_learning_rate,
                   initial_neighbor_radius,
                   dataset_percentage,
                   min_dataset_size,
                   max_iter,
                   training_type="batch"):

        # Initializing parameters:
        self.initial_learning_rate = initial_learning_rate
        self.initial_neighbor_radius = initial_neighbor_radius
        self.epochs = epochs
        self.max_iter = max_iter

        iteration = 0
        can_grow = True
        while can_grow and (iteration < max_iter):
            self.som = SOM(self.nrows,
                           self.ncols,
                           self.parent_dataset,
                           self.initial_learning_rate,
                           self.initial_neighbor_radius,
                           self.epochs,
                           initialization=None)
            self.som.som_map = self.weight_map
            if (training_type == "batch"):
                # print("Training type: {}".format(training_type))
                self.som.som_batch_train(dataset_percentage,
                                         min_dataset_size)
            elif (training_type == "normal"):
                # print("Training type: {}".format(training_type))
                self.som.som_train()
            self.weight_map = self.som.som_map
            self.modify_neuron_weight_matrix()

            iteration += 1
            can_grow = self.allowed_to_grow()
            if can_grow:
                # print("Allowed to grow!")
                self.grow()
                self.nrows, self.ncols = self.current_map_shape
        if can_grow:
            self.map_data_to_neurons()

        return self

    def map_shape(self):
        shape = self.weight_map.shape
        return shape[0], shape[1]

    def modify_neuron_weight_matrix(self):
        """Modify weight matrix of the SOM for the online algorithm.
        Returns
        -------
        np.array
        Weight vector of the SOM after the modification
        """
        for neuron in self.neurons.values():
            weight_vector = self.weight_map[neuron.get_location()]
            neuron.weight_vector = weight_vector

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
                MQE += neuron.compute_quantization_error()
                mapped_neurons += 1

        print("MQE: {}".format(MQE))
        return ((MQE / mapped_neurons) >= (self.map_growing_coefficient *
                                           self.parent_quantization_error)) and \
                                           (changed_neurons > int(np.round(mapped_neurons / 5)))

    def map_data_to_neurons(self):
        for neuron in self.neurons.values():
            neuron.clear_dataset()

        _, data_to_neuron = self.mapping_data_helper(self.parent_dataset)
        neurons = list(self.neurons.values())
        for idx, data_idxs in enumerate(data_to_neuron):
            neurons[idx].replace_dataset(self.parent_dataset[data_idxs, :])

    def find_error_neuron(self):
        quantization_errors = list()
        for neuron in self.neurons.values():
            quantization_error = -np.inf
            if neuron.has_dataset():
                quantization_error = neuron.compute_quantization_error()
            quantization_errors.append(quantization_error)

        error_neuron_index = np.unravel_index(np.argmax(quantization_errors),
                                              shape=self.current_map_shape)
        # error_list_index = np.argmax(quantization_errors)
        # error_neuron = self.neurons[error_neuron_index]
        return self.neurons[error_neuron_index]

    def find_most_dissimilar_neuron(self, error_neuron):
        weight_distances = dict()

        for neuron in self.neurons.values():
            if self.are_neighbors(error_neuron, neuron):
                weight_distance = error_neuron.find_weight_distance_to_other_unit(neuron)
                weight_distances[neuron] = weight_distance

        dissimilar_neuron = max(weight_distances, key=weight_distances.get)
        return dissimilar_neuron

    def grow(self):
        error_neuron = self.find_error_neuron()
        dissimilar_neuron = self.find_most_dissimilar_neuron(error_neuron)

        if self.are_in_same_row(error_neuron, dissimilar_neuron):
            new_row_indices = self.add_column_in_between(error_neuron,
                                                         dissimilar_neuron)
            # print("Same Row")
            self.init_new_weight_vector(new_row_indices,
                                        "horizontal")

        elif self.are_in_same_column(error_neuron, dissimilar_neuron):
            new_column_indices = self.add_row_in_between(error_neuron,
                                                         dissimilar_neuron)
            # print("Same Column")
            self.init_new_weight_vector(new_column_indices,
                                        "vertical")

        else:
            raise RuntimeError("Error neuron and the most dissimilar are not adjacent")

    def add_column_in_between(self, error_neuron, dissimilar_neuron):
        error_col = error_neuron.get_location()[1]
        dissimilar_col = dissimilar_neuron.get_location()[1]
        new_column_idx = max(error_col, dissimilar_col)
        map_rows, map_cols = self.current_map_shape

        # print("Map shape: {}".format(self.current_map_shape))
        # new_column_idx = 2 then [(0, 2), (1, 2), (2, 2)]
        new_line_idx = [(row, new_column_idx) for row in range(map_rows)]
        # print("Adding column")

        for row in range(map_rows):
            for col in reversed(range(new_column_idx, map_cols)):
                new_idx = (row, col + 1)
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
        self.current_map_shape = (self.weight_map.shape[0],
                                  self.weight_map.shape[1])
        # print("Current neurons map shape: {}".format(self.current_map_shape))
        return new_line_idx

    def add_row_in_between(self, error_neuron, dissimilar_neuron):
        """
        Adds an entire row between the error neuron and the dissimilar
        neuron
        """
        error_row = error_neuron.get_location()[0]
        dissimilar_row = dissimilar_neuron.get_location()[0]
        new_row_idx = max(error_row, dissimilar_row)
        map_rows, map_cols = self.current_map_shape

        # print("Map shape: {}".format(self.current_map_shape))
        # new_row_idx = 2 then [(2, 0), (2, 1), (2, 2)]
        new_line_idx = [(new_row_idx, col) for col in range(map_cols)]
        # print("new_row_line_idx: {}".format(new_line_idx))
        # print("Adding row")

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
        self.current_map_shape = (self.weight_map.shape[0],
                                  self.weight_map.shape[1])
        # print("Current neurons map shape: {}".format(self.current_map_shape))
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
        adjacent_neuron_idxs = list()
        if direction == "horizontal":
            adjacent_neuron_idxs = [(row, col - 1), (row, col + 1)]

        elif direction == "vertical":
            adjacent_neuron_idxs = [(row - 1, col), (row + 1, col)]

        # print("Adjacent neuron indices: {}".format(adjacent_neuron_idxs))
        return adjacent_neuron_idxs

    def mean_weight_vector(self, neuron_idxs):
        weight_vector = np.zeros(shape=self.num_input_features,
                                 dtype=np.float32)

        for adjacent_idx in neuron_idxs:
            weight_vector += 0.5 * self.neurons[adjacent_idx].weight_vector
        return weight_vector

    @staticmethod
    def are_neighbors(first_neuron, second_neuron):
        return np.linalg.norm(np.asarray(first_neuron.get_location()) -
                              np.asarray(second_neuron.get_location()),
                              ord=1) == 1

    @staticmethod
    def are_in_same_row(first_neuron, second_neuron):
        """Checks whether the two neurons are in the same row.
        If they are in same row, the difference would be equal to zero.
        """
        first_neuron_row_value = first_neuron.get_location()[0]
        second_neuron_row_value = second_neuron.get_location()[0]

        return abs(first_neuron_row_value - second_neuron_row_value) == 0

    @staticmethod
    def are_in_same_column(first_neuron, second_neuron):
        """
        Checks whether the two neurons are in the same column.
        If they are in same column, the difference would be equal to zero.
        """
        first_neuron_col_value = first_neuron.get_location()[1]
        second_neuron_col_value = second_neuron.get_location()[1]

        return abs(first_neuron_col_value - second_neuron_col_value) == 0


class Growing_RSOM(Growing_SOM):
    def __init__(self,
                 initial_map_size,
                 map_growing_coefficient,
                 weight_map,
                 parent_dataset,
                 parent_quantization_error,
                 neuron_creator):
        """Constructor for Growing_SOM Class."""
        Growing_SOM.__init__(self,
                             initial_map_size=initial_map_size,
                             map_growing_coefficient=map_growing_coefficient,
                             weight_map=weight_map,
                             parent_dataset=parent_dataset,
                             parent_quantization_error=parent_quantization_error,
                             neuron_creator=neuron_creator)

        # Recurrent SOM
        self.differences = np.zeros(self.weight_map.shape)
        self.alpha = 0
        self.num_cycle = 0
        self.num_repeat = 0

    def grsom_train(self,
                    epochs,
                    initial_learning_rate,
                    initial_neighbor_radius,
                    num_cycle,
                    num_repeat,
                    alpha,
                    max_iter,
                    training_type="batch"):
        self.epochs = epochs
        self.initial_learning_rate = initial_learning_rate
        self.initial_neighbor_radius = initial_neighbor_radius
        self.num_cycle = num_cycle
        self.num_repeat = num_repeat
        self.alpha = alpha
        self.max_iter = max_iter

        self.time_constant = self.epochs / np.log(self.initial_neighbor_radius)
        self.differences = np.zeros(self.weight_map.shape)

        # print("cycle: {}, repeat: {}, alpha: {}".format(self.num_cycle,
        #                                                 self.num_repeat,
        #                                                 self.alpha))
        iteration = 0
        can_grow = True
        while can_grow and (iteration < max_iter):
            self.rsom = RSOM(self.nrows,
                             self.ncols,
                             self.parent_dataset,
                             self.initial_learning_rate,
                             self.initial_neighbor_radius,
                             self.epochs,
                             initialization=None)
            self.rsom.som_map = self.weight_map
            if (training_type == "batch"):
                # print("Training type: {}".format(training_type))
                self.rsom.rsom_batch_train(self.epochs,
                                           self.num_cycle,
                                           self.num_repeat,
                                           self.alpha)
            elif (training_type == "normal"):
                # print("Training type: {}".format(training_type))
                self.rsom.rsom_train(self.epochs,
                                     self.num_cycle,
                                     self.num_repeat,
                                     self.alpha)
            self.weight_map = self.rsom.som_map
            self.modify_neuron_weight_matrix()
            
            iteration += 1
            can_grow = self.allowed_to_grow()
            if can_grow:
                # print("Allowed to grow!")
                self.grow()
                self.nrows, self.ncols = self.current_map_shape
        if can_grow:
            self.map_data_to_neurons()

        return self


class GHSOM():
    def __init__(self,
                 input_dataset,
                 map_growing_coefficient,
                 hierarchical_growing_coefficient,
                 initial_learning_rate,
                 initial_neighbor_radius,
                 growing_metric="qe",
                 training_type="batch"):
        """Growing Hierarchical Self Organizing Map Class"""
        self.input_dataset = input_dataset
        self.num_input_features = self.input_dataset.shape[1]

        self.initial_learning_rate = initial_learning_rate
        self.initial_neighbor_radius = initial_neighbor_radius

        self.map_growing_coefficient = map_growing_coefficient
        self.hierarchical_growing_coefficient = hierarchical_growing_coefficient
        self.growing_metric = growing_metric
        self.neuron_creator = Neuron_Creator(self.hierarchical_growing_coefficient,
                                             self.growing_metric)

        self.dataset_percentage = 0.5
        self.min_dataset_size = 20
        self.max_iter = 100
        self.training_type = training_type

        self.zero_neuron = None

    def calc_initial_random_weights(self):
        random_generator = np.random.RandomState(None)
        random_weights = np.zeros(shape=(2, 2, self.num_input_features))

        for location in np.ndindex(2, 2):
            random_data_index = random_generator.randint(len(self.input_dataset))
            random_data_item = self.input_dataset[random_data_index]
            random_weights[location] = random_data_item

        return random_weights

    def create_new_GSOM(self,
                        parent_quantization_error,
                        parent_dataset,
                        weight_map):
        return Growing_SOM((2, 2),
                           self.map_growing_coefficient,
                           weight_map,
                           parent_dataset,
                           parent_quantization_error,
                           self.neuron_creator)

    def create_som_zero_neuron(self):
        self.zero_neuron = self.neuron_creator.zero_neuron(self.input_dataset)
        self.zero_neuron.child_map = self.create_new_GSOM(
            self.neuron_creator.zero_quantization_error,
            self.zero_neuron.input_dataset,
            self.calc_initial_random_weights())

        return self.zero_neuron

    def ghsom_train(self,
                    epochs=15,
                    dataset_percentage=0.50,
                    min_dataset_size=20,
                    max_iter=100):
        self.epochs = epochs
        self.dataset_percentage = dataset_percentage
        self.min_dataset_size = min_dataset_size
        self.max_iter = max_iter

        self.zero_neuron = self.train()

        return self.zero_neuron

    def train(self):

        self.zero_neuron = self.create_som_zero_neuron()

        neurons_to_train_queue = Queue()
        neurons_to_train_queue.put(self.zero_neuron)

        pool = Pool(processes=None)
        while not neurons_to_train_queue.empty():
            size = min(neurons_to_train_queue.qsize(), pool._processes)
            growing_maps = dict()

            for i in range(size):
                neuron = neurons_to_train_queue.get()

                growing_maps[neuron] = (pool.apply_async(neuron.child_map.gsom_train,
                                                         (self.epochs,
                                                          self.initial_learning_rate,
                                                          self.initial_neighbor_radius,
                                                          self.dataset_percentage,
                                                          self.min_dataset_size,
                                                          self.max_iter,
                                                          self.training_type)))
                    
            for neuron in growing_maps:
                growing_map = growing_maps[neuron].get()
                neuron.child_map = growing_map

                neurons_to_expand = filter(lambda _neuron: _neuron.needs_child_map(),
                                           growing_map.neurons.values())

                for _neuron in neurons_to_expand:
                    _neuron.child_map = self.create_new_GSOM(
                        _neuron.compute_quantization_error(),
                        _neuron.input_dataset,
                        self.assign_weights(_neuron.get_location(),
                                            growing_map.weight_map))
                    neurons_to_train_queue.put(_neuron)

        return self.zero_neuron

    def assign_weights(self, parent_location, weight_map):
        child_weights = np.zeros(shape=(2, 2, self.num_input_features))
        stencil = self.generate_kernal_stencil(parent_location)
        for child_location in np.ndindex(2, 2):
            child_location = np.asarray(child_location)

            mask = self.filter_out_of_bound_position(child_location,
                                                     stencil,
                                                     [weight_map.shape[0],
                                                      weight_map.shape[1]])
            weight = np.mean(self.element_from_position_list(weight_map,
                                                             mask),
                             axis=0)
            child_weights[child_location] = weight

        return child_weights

    def filter_out_of_bound_position(self,
                                     child_location,
                                     stencil,
                                     map_shape):
        return np.asarray(list(filter(lambda pos: self.check_position(pos,
                                                                      map_shape),
                                      stencil + child_location)))

    @staticmethod
    def element_from_position_list(matrix, positions_list):
        return matrix[positions_list[:, 0], positions_list[:, 1]]

    @staticmethod
    def generate_kernal_stencil(parent_location):
        row, col = parent_location
        return np.asarray([
            (r, c)
            for r in range(row - 1, row + 1)
            for c in range(col - 1, col + 1)
        ])

    def check_position(self, position, map_shape):
        row, col = position
        map_rows, map_cols = map_shape
        return (row >= 0 and col >= 0) and \
            (row < map_rows and col < map_cols)



class GHRSOM(GHSOM):
    def __init__(self,
                 input_dataset,
                 map_growing_coefficient,
                 hierarchical_growing_coefficient,
                 initial_learning_rate,
                 initial_neighbor_radius,
                 growing_metric,
                 training_type="batch"):
        
        GHSOM.__init__(self,
                       input_dataset=input_dataset,
                       map_growing_coefficient=map_growing_coefficient,
                       hierarchical_growing_coefficient=hierarchical_growing_coefficient,
                       initial_learning_rate=initial_learning_rate,
                       initial_neighbor_radius=initial_neighbor_radius,
                       growing_metric=growing_metric,
                       training_type=training_type)
        # print("Training type: {}".format(training_type))

        self.alpha = 0.7
        self.epochs = 10
        self.num_cycle = 5
        self.num_repeat = 2
        self.zero_neuron = None

    def ghrsom_train(self,
                     epochs=15,
                     num_cycle=5,
                     num_repeat=2,
                     alpha=0.7,
                     max_iter=25):
        self.epochs = epochs
        self.num_cycle = num_cycle
        self.num_repeat = num_repeat
        self.alpha = alpha

        self.zero_neuron = self.train()
        
        return self.zero_neuron

    def create_new_GRSOM(self,
                         parent_quantization_error,
                         parent_dataset,
                         weight_map):
        return Growing_RSOM((2, 2),
                            self.map_growing_coefficient,
                            weight_map,
                            parent_dataset,
                            parent_quantization_error,
                            self.neuron_creator)

    def create_rsom_zero_neuron(self):
        self.zero_neuron = self.neuron_creator.zero_neuron(self.input_dataset)
        print()
        self.zero_neuron.child_map = self.create_new_GRSOM(
            self.neuron_creator.zero_quantization_error,
            self.zero_neuron.input_dataset,
            self.calc_initial_random_weights())

        return self.zero_neuron

    def train(self):

        self.zero_neuron = self.create_rsom_zero_neuron()

        neurons_to_train_queue = Queue()
        neurons_to_train_queue.put(self.zero_neuron)

        pool = Pool(processes=None)
        while not neurons_to_train_queue.empty():
            size = min(neurons_to_train_queue.qsize(), pool._processes)
            growing_maps = dict()

            for i in range(size):
                neuron = neurons_to_train_queue.get()

                growing_maps[neuron] = (pool.apply_async(neuron.child_map.grsom_train,
                                                         (self.epochs,
                                                          self.initial_learning_rate,
                                                          self.initial_neighbor_radius,
                                                          self.num_cycle,
                                                          self.num_repeat,
                                                          self.alpha,
                                                          self.max_iter,
                                                          self.training_type)))
                    
            for neuron in growing_maps:
                growing_map = growing_maps[neuron].get()
                neuron.child_map = growing_map

                neurons_to_expand = filter(lambda _neuron: _neuron.needs_child_map(),
                                           growing_map.neurons.values())

                for _neuron in neurons_to_expand:
                    _neuron.child_map = self.create_new_GRSOM(
                        _neuron.compute_quantization_error(),
                        _neuron.input_dataset,
                        self.assign_weights(_neuron.get_location(),
                                            growing_map.weight_map))
                    neurons_to_train_queue.put(_neuron)

        return self.zero_neuron
