import numpy as np

##### fix these
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()
#####
def sigmoid(z):
    return 1/(1+np.exp(-1*z))

def softmax(z):
    z_exp = np.exp(z)
    return z_exp/z_exp.sum(axis=1, keepdims=True) if len(z_exp.shape)==2 else z_exp/sum(z_exp)

class MyNN:
    def __init__(self, config=None):
        self.layers = []

    def add_layer(self, weight_array, bias_vector, activation_function, activation_derivative=None):
        self.layers.append(MyLayer(weight_array, bias_vector, activation_function, activation_derivative))

    def forward(self, data, stop_at=None, return_all_activations=False):
        # initial activations are the input data
        activations = data
        all_activations = [activations]
        # allow forward to stop early for debuging
        stop_at = min(len(self.layers), stop_at) if stop_at is not None else len(self.layers)
        for i in range(stop_at):
            layer = self.layers[i]
            # the output of one layer is the input for the next layer
            activations = layer.activation_function(activations.dot(layer.weight_array) + layer.bias_vector)
            all_activations.append(activations)
        return all_activations if return_all_activations else activations

    def backward(self, data, labels, activation_list):
        # batch gradient descent
        # get in reverse order since we are walking backwards through the network
        weights_list = [layer.weight_array for layer in self.layers][::-1]

        # calculate the deltas
        # walk through layers backwards
        for index, layer in enumerate(self.layers[::-1]):
            # activations from current layer
            a = activation_list[index]
            if not index:
                inputs = {'a': a, 'labels': labels}
                delta_list = [layer.activation_derivative(**inputs)]
            else:
                previous_delta = delta_list[index - 1]
                # weighting matrix from previous layer
                w = weights_list[index - 1]
                # calculate the new delta
                delta = previous_delta.dot(w.T) * layer.activation_derivative(a)
                delta_list.append(delta)

        # return bias_gradients in reverse order to match what update_weights expects
        bias_gradients = [delta.sum(axis=0) for delta in delta_list][::-1]

        weight_gradients = []
        for index, delta in enumerate(delta_list):
            # activation_list has activations in reverse order and is one element bigger than delta_list
            a_previous = activation_list[index + 1]
            weight_gradients.append(a_previous.T.dot(delta))

        weight_gradients = weight_gradients[::-1]

        return ({  'bias_gradients': bias_gradients
                 , 'weight_gradients': weight_gradients
                }
               )

    def update_weights(self, learning_rate, gradients):
        self.layers[0].weight_array = self.layers[0].weight_array + learning_rate*gradients['weight_gradients'][0]
        self.layers[1].weight_array = self.layers[1].weight_array + learning_rate*gradients['weight_gradients'][1]

        self.layers[0].bias_vector = self.layers[0].bias_vector + learning_rate*gradients['bias_gradients'][0]
        self.layers[1].bias_vector = self.layers[1].bias_vector + learning_rate*gradients['bias_gradients'][1]

    def train(self, data, labels, learning_rate, iterations):
        for i in range(iterations):
            # get in reverse order since we are walking backwards through the network
            activation_list = self.forward(data, return_all_activations=True)[::-1]
            output = activation_list[0]
            # get the gradients
            gradients = self.backward(data, labels, activation_list)
            # use the gradients to updates the weights
            self.update_weights(learning_rate, gradients)
            c = cost(labels, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print("cost:", c, "classification_rate:", r)


class MyLayer:
    def __init__(self, weight_array, bias_vector, activation_function, activation_derivative=None):
        self.weight_array = weight_array
        self.bias_vector = bias_vector
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
