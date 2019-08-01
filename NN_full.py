import math as cm
import numpy as np


def sigmoid(x):
    return 1 / (1 + cm.exp(-x))


def dot(K, L):
    if len(K) != len(L):
        return 0
    return sum(i[0] * i[1] for i in zip(K, L))


def d_sigmoid(x):
    dfx = sigmoid(x) * (1 - sigmoid(x))
    return dfx

def d_f_sigmoid(sigmoid):
    d = sigmoid*(1-sigmoid)
    return d


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
    def __init__(self, dim_inputs, dim_outputs, n_layers, n_neurons):
        # We initialize the input vector
        self.inputs = []
        for i in range(dim_inputs):
            self.inputs.append(np.random.random())
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
        self.layers.append(layer(self.layers[n_layers - 1].neurons, dim_outputs))

    def predict(self, inputs):
        self.outputs_p = []
        for i, elementi in enumerate(self.layers):
            for elementj in elementi.neurons:
                # For the fist layer we input the system inputs.
                if i == 0:
                    elementj.feedforward(inputs)
                # For the rest of layers we input the previous layer values.
                else:
                    # We make a list of each neuron activation values from the previous layer.
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
    # The input is a list of all the data set, if we have 2 outputs, and 3 samples
    # real_outputs
    def mse_loss(self, target, output):
        mse = 0.5*(target-output)**2
        return mse

    # Derivate of our error, in order to minimize it.
    def d_mse_loss(self, target, output):
        d_mse = -(target-output)
        return d_mse

    def backpropagation(self, learning_rate, input, outputs_p, outputs_r):
        # Derivates Pre Calculations
        d_out_d_in = []
        d_in_d_weight = []
        for i, layer in enumerate(self.layers[::-1]):
            # We do it until we reach last layer.
            d_out_d_in_layer = []
            d_in_d_weight_layer = []
            for j, neuron in enumerate(layer.neurons):
                d_out_d_in_layer.append(d_f_sigmoid(neuron.last_activated))
                d_in_d_weight_layer_neuron = []
                if i < len(self.layers) - 1:
                    for k, neuron_ant in enumerate(self.layers[::-1][i + 1].neurons):
                        d_in_d_weight_layer_neuron.append(neuron_ant.last_activated)
                else:
                    for element in input:
                        d_in_d_weight_layer_neuron.append(element)
                d_in_d_weight_layer.append(d_in_d_weight_layer_neuron)
            d_out_d_in.append(d_out_d_in_layer)
            d_in_d_weight.append(d_in_d_weight_layer)
        # Errors respect the input of the actual node.
        d_Ei_outi = []
        for i, layer in enumerate(self.layers[::-1]):
            d_Ei_outi_layer = []
            for j, neuron in enumerate(layer.neurons):
                if i == 0:
                    d_Ei_outi_layer.append(self.d_mse_loss(outputs_r[j], outputs_p[j]))
                else:
                    sum = 0
                    for k, neuron_ant in enumerate(self.layers[::-1][i-1].neurons):
                        sum = sum + d_Ei_outi[i - 1][k] * d_out_d_in[i - 1][k] * neuron_ant.w[j]
                    d_Ei_outi_layer.append(sum)
            d_Ei_outi.append(d_Ei_outi_layer)
        # Deltas
        deltas = []
        deltas_b = []
        for i, layer in enumerate(d_Ei_outi):
            deltas_layer = []
            deltas_b_layer = []
            for j, derror in enumerate(layer):
                deltas_layer_neuron = []
                deltas_b_layer.append(derror * d_out_d_in[i][j])
                for dweight in d_in_d_weight[i][j]:
                    deltas_layer_neuron.append(derror * d_out_d_in[i][j] * dweight)
                deltas_layer.append(deltas_layer_neuron)
            deltas_b.append(deltas_b_layer)
            deltas.append(deltas_layer)

        # New weights:
        for i, dlayer in enumerate(deltas):
            for j, dneuron in enumerate(dlayer):
                self.layers[::-1][i].neurons[j].b = self.layers[::-1][i].neurons[j].b - learning_rate * deltas_b[i][j]
                for h, dneuroninp in enumerate(dneuron):
                    self.layers[::-1][i].neurons[j].w[h] = self.layers[::-1][i].neurons[j].w[h] - learning_rate*dneuroninp

    def train_SGD(self, inputs, outputs, learning_rate):
        for i, input in enumerate(inputs):
            self.backpropagation(learning_rate, input, self.predict(input), outputs[i])

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
Number_neurons = [4, 2]
# x,y
def create_data(number): # 4 Quadrants data
    inputs = []
    outputs = []
    for j in range(4):
        for i in range(number):
            if j == 0:
                a = 1
                b = 1
            elif j == 1:
                a = -1
                b = 1
            elif j == 2:
                a = -1
                b = -1
            elif j == 3:
                a = 1
                b = -1
            i1 = np.random.random()*a
            i2 = np.random.random()*b
            inputs.append([i1, i2])
    np.random.shuffle(inputs)
    for element in inputs:
        if element[0] >= 0:
            if element[1] >= 0:
                outputs.append([1])
            else:
                outputs.append([0])
        else:
            if element[1] >= 0:
                outputs.append([0])
            else:
                outputs.append([1])
    return inputs, outputs

inputs, outputs = create_data(1)
dim_inputs = len(inputs[0])
dim_outputs = len(outputs[0])
epochs = 100
l_rate = 0.1
nn = NeuralNetwork(dim_inputs, dim_outputs, Number_layers, Number_neurons)


print('Initial prediction for [-0.5,0.8] should be 0:', nn.predict([-0.5, 0.8]))

for i in range(epochs):
    inputs, outputs = create_data(1000)
    nn.train_SGD(inputs, outputs, l_rate)

print('Trained prediction for [-0.5,0.8] should be 0:', nn.predict([-0.5, 0.8]))
print('Trained prediction for [-0.5,-0.2] should be 1:', nn.predict([-0.5, -0.2]))
print('Trained prediction for [0.1,-0.2] should be 0:', nn.predict([0.1, -0.2]))

