import numpy as np

def classification_accuracy(actual_labels, predicted_labels):
    return np.mean(actual_labels == predicted_labels)

def cross_entropy(y_indicator, nn_output):
    tot = y_indicator * np.log(nn_output)
    return tot.sum()

def sigmoid(z):
    return 1/(1+np.exp(-1*z))

def softmax(z):
    z_exp = np.exp(z)
    return z_exp/z_exp.sum(axis=1, keepdims=True) if len(z_exp.shape)==2 else z_exp/sum(z_exp)

class MyNN:
    def __init__(self, config=None):
        self.layers = []

    def add_layer(self, weight_matrix, bias_vector, activation_function, activation_derivative=None):
        self.layers.append(MyLayer(weight_matrix, bias_vector, activation_function, activation_derivative))

    def forward(self, data, stop_at=None, return_all_activations=False):
        # initial activations are the input data
        activations = data
        all_activations = [activations]
        # allow forward to stop early for debuging
        stop_at = min(len(self.layers), stop_at) if stop_at is not None else len(self.layers)
        for i in range(stop_at):
            layer = self.layers[i]
            # the output of one layer is the input for the next layer
            activations = layer.activation_function(activations.dot(layer.weight_matrix) + layer.bias_vector)
            all_activations.append(activations)
        return all_activations if return_all_activations else activations

    def backward(self, data, y_indicator, all_activations):
        # batch gradient descent
        # put in reverse order since we are walking backwards through the network
        all_activations = all_activations[::-1]
        # get in reverse order since we are walking backwards through the network
        weight_matrices = [layer.weight_matrix for layer in self.layers][::-1]

        # calculate the deltas
        # walk through layers backwards
        for index, layer in enumerate(self.layers[::-1]):
            # activations from current layer
            a = all_activations[index]
            # handle the first layer specially because the gradient has two inputs.
            if not index:
                inputs = {'a': a, 'y_indicator': y_indicator}
                deltas = [layer.activation_derivative(**inputs)]
            else:
                previous_delta = deltas[index - 1]
                # weighting matrix from previous layer
                w = weight_matrices[index - 1]
                # calculate the new delta
                delta = previous_delta.dot(w.T) * layer.activation_derivative(a)
                deltas.append(delta)

        # return bias_gradients in forward order to match what update_weights expects
        bias_gradients = [delta.sum(axis=0) for delta in deltas][::-1]

        weight_gradients = []
        for index, delta in enumerate(deltas):
            # all_activations has activations in reverse order and is one element larger than deltas
            a_previous = all_activations[index + 1]
            weight_gradients.append(a_previous.T.dot(delta))

        # return weight_gradients in forward order to match what update_weights expects
        weight_gradients = weight_gradients[::-1]

        return {'bias_gradients': bias_gradients, 'weight_gradients': weight_gradients}


    def update_weights(self, learning_rate, gradients):
        for index, layer in enumerate(self.layers):
            self.layers[index].weight_matrix = layer.weight_matrix + learning_rate*gradients['weight_gradients'][index]
            self.layers[index].bias_vector = layer.bias_vector + learning_rate*gradients['bias_gradients'][index]

    def train(self, train, yInd_train, learning_rate, iterations):
        for i in range(iterations):
            all_activations = self.forward(train, return_all_activations=True)
            nn_output = all_activations[-1]
            # get the gradients
            gradients = self.backward(train, yInd_train, all_activations)
            # use the gradients to updates the weights
            self.update_weights(learning_rate, gradients)
            cost = cross_entropy(yInd_train, nn_output)
            actual_labels = np.argmax(yInd_train, axis=1)
            predicted_labels = np.argmax(nn_output, axis=1)
            accuracy = classification_accuracy(actual_labels, predicted_labels)
            print("cost:", cost, "accuracy:", accuracy)


class MyLayer:
    def __init__(self, weight_matrix, bias_vector, activation_function, activation_derivative=None):
        self.weight_matrix = weight_matrix
        self.bias_vector = bias_vector
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
