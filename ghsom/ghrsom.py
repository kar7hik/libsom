import numpy as np
from matplotlib import pyplot as plt
from queue import Queue
from multiprocessing import Pool


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
        self.zero_unit_quantization_error = zero_quantization_error
        self.child_map = None
        self.input_dataset = None
        
        self.num_input_features = self.weight_vector.shape[0]

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

    def find_distance_for_dataset(self, dataset):
        """
        Returns the Euclidean norm between neuron's weight vector and the \
        whole input dataset
        """
        distance = np.linalg.norm(np.subtract(self.weight_vector,
                                              dataset),
                                  ord=2, axis=1)
        return distance

    def activation(self, datapoint):
        activation = np.linalg.norm(np.subtract(datapoint,
                                                self.weight_vector),
                                    ord=2)
        return activation

    def compute_quantization_error(self, growing_metric=None):
        """
        Returns the Quantization error
        """
        if growing_metric is None:
            growing_metric = self.growing_metric
        distance_from_whole_dataset = self.find_weight_distance_to_other_unit(self.input_dataset)
        quantization_error = distance_from_whole_dataset.sum()

        if (self.growing_metric == "mqe"):
            quantization_error /= self.input_dataset.shape[0]
            return quantization_error
        else:
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
        Returns the Boolean value. Checks whether the current neuron has any data set
        """
        return len(self.input_dataset) != 0

    def init_empty_dataset(self):
        """
        Resets the current input dataset to an empty dataset with the dimension of \
        0 and the number of input data features.
        """
        return np.empty(shape=(0,
                               self.num_input_features),
                        dtype=np.float32)

    def replace_dataset(self, dataset):
        """
        Replaces the current input dataset of the neuron with the given dataset
        """
        self.input_dataset = dataset

    def clear_dataset(self):
        """
        Clears the current input dataset of the neuron
        """
        self.input_dataset = self.init_empty_dataset()



class Neuron_Creator():
    zero_unit_quantization_error = None
    def __init__(self, hierarchical_growing_coefficient, growing_metric):
        # TODO: Better name for tau_2
        self.hierarchical_growing_coefficient = hierarchical_growing_coefficient
        self.growing_metric = growing_metric

    def new_neuron(self, neuron_location, weight_vector):
        assert self.zero_unit_quantization_error is not None, \
            "Unit zero's quantization error has not been set yet."
        return Neuron(neuron_location,
                      weight_vector,
                      self.growing_metric,
                      self.hierarchical_growing_coefficient,
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
                 map_growing_coefficient,
                 weight_map,
                 parent_dataset,
                 parent_quantization_error,
                 neuron_creator,
                 native_update=True):
        """Constructor for Growing_SOM Class."""
        self.initial_map_size = initial_map_size
        self.weight_map = weight_map
        self.parent_dataset = parent_dataset
        self.parent_quantization_error = parent_quantization_error
        self.num_input_features = parent_dataset.shape[1]
        self.map_growing_coefficient = map_growing_coefficient

        self.neuron_creator = neuron_creator
        self.neurons = self.create_neurons()

        self.current_som_map = weight_map
        self.current_map_shape = self.initial_map_size

        # self.node_list = np.array(list(
        #     itertool.product(range(self.current_map_shape[0]),
        #                      range(self.current_map_shape[1]))),
        #                           dtype=int)

        self.som = None
        self.init_learning_rate = None
        self.init_radius = None
        self.time_constant = None


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

    def train(self,
              epochs,
              max_iteration,
              init_learning_rate,
              init_radius,
              dataset_percentage,
              min_dataset_size):
        iteration = 0
        can_grow = True

        while can_grow and (iteration < max_iteration):
            self.neurons_training(epochs,
                                  init_learning_rate,
                                  init_radius,
                                  dataset_percentage,
                                  min_dataset_size)
            iteration += 1
            can_grow = self.allowed_to_grow()

            if can_grow:
                self.grow()

        if can_grow:
            self.map_data_to_neurons()

        return self

    def neurons_training(self,
                         epochs,
                         init_learning_rate,
                         init_radius,
                         dataset_percentage,
                         min_dataset_size):
        lr = init_learning_rate
        neighbor_radius = init_radius

        for iteration in range(epochs):
            self.som_batch_update(self.)
    
    def decay_radius(self, current_iter):
        return self.init_radius * np.exp(-current_iter / self.time_constant)

    def decay_learning_rate(self, current_iter):
        return self.init_learning_rate * \
            np.exp(-current_iter / self.num_iteration)

    def calculate_influence(self, w_dist, neighbor_radius):
        pseudogaussian = np.exp(-np.divide(np.power(w_dist, 2),
                                           (2 * np.power(neighbor_radius, 2))))
        return pseudogaussian

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

    def som_batch_update(self,
                         training_data,
                         learning_rate,
                         neighbor_radius):
        bmu_index = self.get_bmu(training_data)
        w_dist = np.linalg.norm(self.node_list_ - bmu_index, axis=1)
        neighbor_influence = self.calculate_influence(w_dist,
                                                      neighbor_radius).reshape(
                                                          self.nrows,
                                                          self.ncols,
                                                          1)
        self.som_map = self.modify_weight_matrix(learning_rate,
                                                 neighbor_influence,
                                                 training_data)
        # print(self.som_map)

    def batch_training_data(self, dataset_percentage, min_size, seed):
        dataset_size = self.input_len
        if dataset_size <= min_size:
            iterator = range(dataset_size)
        else:
            iterator = range(int(ceil(dataset_size * dataset_percentage)))

        random_generator = np.random.RandomState(seed)
        for _ in iterator:
            yield self.input_data[random_generator.randint(dataset_size)]
        
    def batch_train_som(self, dataset_percentage,
                        min_size, seed):
        lr = self.init_learning_rate
        nr = self.init_radius

        for i in range(self.num_iteration):
            for data in self.batch_training_data(dataset_percentage,
                                                 min_size,
                                                 seed):
                self.som_batch_update(data, lr, nr)

            nr = self.decay_radius(i)
            lr = self.decay_learning_rate(i)


    def som_update(self, datapoint, learning_rate, neighbor_radius):
        bmu_index = self.get_bmu(datapoint)
        w_dist = np.linalg.norm(self.node_list_ - bmu_index, axis=1)
        neighbor_influence = self.calculate_influence(w_dist,
                                                      neighbor_radius).reshape(
                                                          self.nrows,
                                                          self.ncols,
                                                          1)
        self.som_map = self.modify_weight_matrix(learning_rate,
                                                 neighbor_influence,
                                                 datapoint)
            
    def train_som(self):
        dataset_percentage = 1.0
        min_size = self.input_len
        seed = None

        lr = self.init_learning_rate
        nr = self.init_radius

        for i in range(self.num_iteration):
            for data in self.batch_training_data(dataset_percentage,
                                                 min_size,
                                                 seed):
                self.som_update(data, lr, nr)
            nr = self.decay_radius(i)
            lr = self.decay_learning_rate(i)
    
    
# Testing Section #############################################################
hierarchical_growing_coefficient = 0.001
empty_weight_vector = np.zeros(shape=(0, 3), dtype=float)

raw_data = np.random.randint(0, 255, (1000, 3))
data = raw_data
col_maxes = raw_data.max(axis=0)
input_dataset = raw_data / col_maxes[np.newaxis, :]



neuron_creator = Neuron_Creator(hierarchical_growing_coefficient,
                                "qe")

