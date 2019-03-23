import math as cm
import numpy as np

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + cm.exp(-x))


def dot(K, L):
    if len(K) != len(L):
        return 0
    return sum(i[0] * i[1] for i in zip(K, L))


def d_sigmoid(x):
    dfx = sigmoid(x) * (1 - sigmoid(x))
    return dfx


# Each neuron has it's weights and it's bias.
# It has as many weights as inputs, and 1 bias; Defined in layer().
class neuron():
    def __init__(self, weight, bias):
        self.w = weight
        self.b = bias
        self.last_activated = 0

    def activation(self, x):
        return sigmoid(x + self.b)

    # For each element in the input x, x0w0 + x1w1 + x2w1... it's the dot product
    # self.last_activated has the value of the activated neuron.
    def feedforward(self, x):
        weighter = dot(x, self.w)
        self.last_activated = self.activation(weighter)
        return self.last_activated


# Create a layer with as many weights as inputs, and as many neurons and biases as n_neurons
# Each layer it's composed of neurons: [neuron_1, neuron_2, neuron_3..., neuron_n]
# Each neuron is composed of m weights, being m = number of inputs to the layer, and one bias (one for each neuron);
# neuron 1 has its weights wn1 = [w'11, w'21, ..., w'n1] ' is for the layer 1, '' for layer 2, etc;
# 11 is for input 1 neuron 1, 21 for input 2 neuron 1.
class layer():
    def __init__(self, inputs, n_neurons):
        self.neurons = []
        # For each neuron we are creating in the layer:
        for i in range(n_neurons):
            w = []
            # We create a weight for each input into the neuron we are creating.
            for element in range(len(inputs)):
                w.append(np.random.random())
            neuron_i = neuron(w, np.random.random())
            # Once we create all the input weights for the neuron we are creating, we input
            # it to the list of weights to the our list of neurons for our layer.
            self.neurons.append(neuron_i)


class NeuralNetwork():

    def __init__(self, n_inputs, n_outputs, n_layers, n_neurons):
        # We initialize the input vector
        self.inputs = [50, 4, 1]
        # for i in range(n_inputs):
            # self.inputs.append(np.random.random())
        #Initialize output real vector
        self.outputs_r = [20, 5]
        # for i in range(n_outputs):
            # self.outputs_r.append(np.random.random())
        # Initialize output prediction vector
        self.outputs_p = []
        self.layers = []
        # Create a NN from inputs to last layer, with all weights and biases, still need to crate weights from last
        # layer to output and output biases.
        # Number of layers does not include output.
        for i in range(n_layers):
            if i == 0:
                # Create first layer coming from inputs.
                self.layers.append(layer(self.inputs, n_neurons[i]))
            else:
                self.layers.append(layer(self.layers[i - 1].neurons, n_neurons[i]))
        # Create output with it's biases and weights, coming from last layer.
        self.layers.append(layer(self.layers[n_layers - 1].neurons, len(self.outputs_r)))

    def predict(self, inputs):
        self.outputs_p = []
        for i, elementi in enumerate(self.layers):
            for elementj in elementi.neurons:
                # For the fist layer we input the system inputs.
                if i == 0:
                    elementj.feedforward(inputs)
                # For the rest of layers we input the previous layer values.
                else:
                    # We make a list of each neuron activation values from teh previous layer.
                    values = []
                    for elementk in self.layers[i - 1].neurons:
                        values.append(elementk.last_activated)
                    # We input the values from the previous layer to the current layer.
                    elementj.feedforward(values)
        # Our output will be the activation values of the last layer.
        for element in self.layers[len(self.layers) - 1].neurons:
            self.outputs_p.append(element.last_activated)
        return self.outputs_p

    # We can calculate the mean squared error (MSE)
    def mse_loss(self):
        mse = 0
        for i, real_out in enumerate(self.outputs_r):
            mse = mse + (real_out - self.outputs_p[i]) ** 2
        mse = mse / len(self.outputs_p)
        return mse

    def train(self, inputs, real_outputs):
        self.predict(inputs)

    def __str__(self):
        representation = 'Neural network layer formed by ' + str(len(self.layers) - 1) + ' hidden layers.\n'
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

        return representation  # what it shows when we call print over an object which is class layer.



Number_layers = 2
# Number of neurons of each hidden layer.
Number_neurons = [4, 2]
number_inputs = 3
number_outputs = 2
nn = NeuralNetwork(number_inputs, number_outputs, Number_layers, Number_neurons)

input = [50, 4, 1]
real_output = [20, 5]
print(nn)
nn.predict(input)