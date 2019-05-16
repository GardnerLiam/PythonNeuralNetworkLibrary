"""
@Author: Liam Gardner
@Date: 5/16/2019
"""
import numpy as np


def sigmoid(x, deriv=False):
    """
    Sigmoid activation function
    :param x: input value
    :param deriv: returns derivative of sigmoid if true
    :return: returns either sigmoid(x) or sigmoid derivative of x
             the values passed through the derivative have already been passed through the sigmoid function.
             Thus, the derivative is x(1-x) rather than sigmoid(x)*(1-sigmoid(x))
    """
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class Layer(object):
    def __init__(self, input_dim, output_dim):
        """
        Layer class, this class holds weights for each layer
        creates weights and biases based on inputs and outputs
        :param input_dim: amount of input neurons
        :param output_dim: amount of output neurons
        """

        self.inputs = input_dim
        self.outputs = output_dim
        self.weight = 2 * np.random.random((input_dim, output_dim)) - 1
        self.bias = 2 * np.random.random((output_dim)) - 1

    def feedForward(self, inputs):
        """
        Passes input matrix through weights and biases
        :param inputs: layer input
        :return: layer output
        """
        return sigmoid(np.dot(inputs, self.weight) + self.bias)


class NeuralNetwork(object):
    def __init__(self, input_dim):
        """
        Neural Network class. Stores a list of layers, the network's error
        :param input_dim: dimensions of the input vectors
        """
        self.input_dim = input_dim
        self.layers = []
        self.error = 0

    def feedForward(self, inputs):
        """
        Return neural network output given input.
        :param inputs: input matrix
        :return: output of neural network
        """
        res = inputs
        for l in self.layers:
            res = l.feedForward(res)
        return res

    def addLayer(self, inputs, outputs):
        """
        add a layer to the neural network
        :param inputs: input neurons
        :param outputs: output neurons
        :return: None
        """
        self.layers.append(Layer(inputs, outputs))

    def backPropagate(self, inputs, outputs, alpha=0.1):
        """
        Backpropagation algorithm. Perform gradient descent and adjust the weights and biases of each layer.
        :param inputs: input matrix
        :param outputs: correct output
        :param alpha: learning rate
        :return: None
        """

        # create a layer output variable which initally stores the input
        # perform the feed forward algorithm with the last value of the layer_output list for each layer.
        # Append the layer's output to layer_output
        layer_outputs = [inputs]
        for l in self.layers:
            layer_outputs.append(l.feedForward(layer_outputs[-1]))

        # initialize the error list with the difference of the output compared to the correct answer.
        # initialize the delta list with the gradient of the last layer in the neural network
        # save the mean squared error of the last layer into the neural network's error variable
        errors = [outputs - layer_outputs[-1]]
        deltas = [errors[-1] * sigmoid(layer_outputs[-1], deriv=True)]
        self.error = np.mean(errors[0] * errors[0])  # save mean square error

        # for every layer in the neural network, starting from last to first:
        #   - Calculate the layer's error and append it to the error list
        #   - Using the newly calculated error and the output of the layer, calculate that layer's gradient
        #   - Move back one layer
        layer_index = len(self.layers) - 1
        while (layer_index > 0):
            errors.append(np.matmul(deltas[-1], self.layers[layer_index].weight.T))
            deltas.append(errors[-1] * sigmoid(layer_outputs[layer_index], deriv=True))
            layer_index -= 1

        # for every layer in the neural network, starting from last to first:
        #   - Calculate the dot product of the previous layer's output (the first hidden layer will use the inputs)
        #       with the gradient of that layer
        #   - Multiply the result of that dot product by the learning rate
        #   - Add the result to the layer's weight
        #   - Vertically sum the gradient of that layer, multiply the result by the learning rate
        #   - Add the result to the layer's bias
        output_index = len(layer_outputs) - 2
        delta_index = 0
        for l in self.layers[::-1]:
            o = np.array([layer_outputs[output_index]]).T
            d = np.array([deltas[delta_index]])
            prod = np.dot(o, d) * alpha
            l.weight += prod
            l.bias += np.sum(deltas[delta_index], axis=0) * alpha
            output_index -= 1
            delta_index += 1