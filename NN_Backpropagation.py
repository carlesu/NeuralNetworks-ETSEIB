import math as cm
import numpy as np



def sigmoid(x):
    return 1/(1 + cm.exp(-x))


def dot(K, L):
   if len(K) != len(L):
      return 0
   return sum(i[0] * i[1] for i in zip(K, L))


# Each neuron has it's weights and it's bias.
# It has as many weights as inputs, and 1 bias; Defined in layer().
class neuron():
    def __init__(self, weight, bias):
        self.w = weight
        self.b = bias


# Create a layer with as many weights as inputs, and as many neurons and biases as n_neurons
# Each layer it's composed of neurons: [neuron_1, neuron_2, neuron_3..., neuron_n]
# Each neuron is composed of m weights, being m = number of inputs to the layer, and one bias (one for each neuron);
# neuron 1 has its weights wn1 = [w'11, w'21, ..., w'n1] ' is for the layer 1, '' for layer 2, etc;
# 11 is for input 1 neuron 1, 21 for input 2 neuron 1.
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
        self.FeedForward()


    def FeedForward(self):
        self.lvalues = []
        self.lvalues.append(self.inputs)
        for i in range(len(self.layers)):
            values = []
            for j in range(len(self.layers[i].neurons)):
                values.append(sigmoid(dot(self.lvalues[i], self.layers[i].neurons[j].w) + self.layers[i].neurons[j].b))
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


#Your work is to generalize the basic neural network we saw at class session.
#You should have a class that initializes with the number of inputs,
# numbers of layers, size of each layer and number of outputs.
#Once instantiated the class should have a feedforward function which
# should calculate the output as a function of the input.
#The class must have all the required structures to accommodate the weights
# and bias of each neuron on each layer ad should be implemented with lists, no numpy yet.


input = [50, 4, 1]
real_output = [20, 5]
Number_layers = 2
# Number of neurons of each hidden layer.
Number_neurons = [4,2]
nn = NeuralNetwork(input, real_output, Number_layers, Number_neurons)
print(nn)