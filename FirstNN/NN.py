import math as cm
import numpy as np


np.random.seed(42)


def sigmoid(x):
    return 1/(1 + cm.exp(-x))


def dot(K, L):
   if len(K) != len(L):
      return 0
   return sum(i[0] * i[1] for i in zip(K, L))


class neuron():
    def __init__(self, weight, bias):
        self.w = weight
        self.b = bias

    def activation(self, x):
        return sigmoid(x + self.b)

    # For each element in the input x, x0w0 + x1w1 + x2w1... it's the dot product
    # self.last_activated has the value of the activated neuron.
    def feedforward(self, x):
        weighter = 0
        for i, element in enumerate(x):
            weighter = weighter + element*self.w[i]
        self.last_activated = self.activation(weighter)
        return self.last_activated


class layer():
    def __init__(self, inputs, n_neurons):
        self.neurons = []
        for i in range(n_neurons):
            w = []
            for element in range(len(inputs)):
                w.append(np.random.random())
            neuron_i = neuron(w, np.random.random())
            self.neurons.append(neuron_i)



class NeuralNetwork():

    def __init__(self, inputs, outputs, n_layers, n_neurons):
        self.inputs = inputs
        self.outputs_p = outputs  # Prediction
        self.outputs_r = outputs  # Real
        self.layers = []
        # We create the first layer values, which will be the input.
        self.lvalues = []
        self.lvalues.append(self.inputs)

        # Create a NN from inputs to last layer, with all weights and biases, still need to cerate weights from last
        # layer to output and output biases.
        # Number of layers include output.
        for i in range(n_layers):
            if i == 0:
                self.layers.append(layer(self.inputs, n_neurons[i]))  # Create inputs and it's weights
            else:
                self.layers.append(layer(self.layers[i-1].neurons, n_neurons[i]))

        # Create output with it's biases and weights, from last layer.
        self.layers.append(layer(self.layers[n_layers-1].neurons, len(self.outputs_r)))

        # We call the feed forward function.
        for i, elementi in enumerate(self.layers):
            values = []
            for elementj in elementi.neurons:
                elementj.feedforward(self.lvalues[i])
                values.append(elementj.last_activated)
            self.lvalues.append(values)

        # The last value of self.lvalues is the output y predicted.
        self.outputs_p = self.lvalues[len(self.layers)]


    def __str__(self):
        representation = 'Neural network layer formed by ' + str(len(self.layers)-1) + ' hidden layers.\n'
        representation = representation + 'Each layer is composed by ' + str(len(self.layers[0].neurons)) \
                         + ' neurons.\n'
        for i, element in enumerate(self.layers):
            if i != (len(self.layers) - 1):
                representation = representation + '\n'
                representation = representation + 'For the layer ' + str(i) + ' we have the neurons:\n'
            else:
                representation = representation + '\n'
                representation = representation + 'For the OUTPUT layer we have the neurons:\n'

            for j, values in enumerate(self.layers[i].neurons):
                representation = representation + 'Neuron ' + str(j) + ':\n'
                representation = representation + '\t\t input weights: ' + str(values.w) + '\n'
                representation = representation + '\t\t bias: ' + str(values.b) + '\n'

        representation = representation + '\n'
        representation = representation + 'The real output is' + str(self.outputs_r) + '.\n'
        representation = representation + 'The predicted output is' + str(self.outputs_p) + '.\n'

        return representation # what it shows when we call print over an object which is class layer.


input = [50, 4, 1]
real_output = [20, 5]
Number_layers = 2
# Number of neurons of each hidden layer.
Number_neurons = [4,2]
nn = NeuralNetwork(input, real_output, Number_layers, Number_neurons)
print(nn)