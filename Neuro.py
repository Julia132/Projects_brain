from numpy import exp, array, random, dot
import pandas as pd
import numpy as np
import csv
class NeuralNetwork():
    training_set_inputs = pd.read_csv("C:/Users/inet/Documents/GitHub/Projects_brain/data_set_2.csv", sep=',',
                                      encoding='latin1',
                                      dayfirst=True,
                                      index_col=None, header=None)
    def __init__(self):
        self.synaptic_weights = np.zeros(self.training_set_inputs.shape)
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)
    training_set_inputs = pd.read_csv("C:/Users/inet/Documents/GitHub/Projects_brain/data_set_2.csv", sep=',', encoding='latin1',
                          dayfirst=True,
                          index_col=None, header=None)
    training_set_outputs = pd.read_csv("C:/Users/inet/Documents/GitHub/Projects_brain/out.csv", sep=',', encoding='latin1',
                          dayfirst=True,
                          index_col=None, header=None).T
    neural_network.train(training_set_inputs, training_set_outputs, 10000)
    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

