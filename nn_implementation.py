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

    def add_layer(self, weight_array, bias_vector, activation_function):
        self.layers.append(MyLayer(weight_array, bias_vector, activation_function))

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

    def backward(self, data, labels):
        # do batch gradient descent
        activation_list = []
        # get all activations since they will all be used to adjust the NN weights
        a0, a1, a2 = self.forward(data, return_all_activations=True)
        w1, w2 = [layer.weight_array for layer in self.layers]

        delta2 = labels - a2
        grad_w2 = a1.T.dot(delta2)
        grad_b2 = delta2.sum(axis=0)

        delta1 = delta2.dot(w2.T)
        grad_w1 = a0.T.dot(delta1 * a1*(1 - a1))
        grad_b1 = (delta1 * a1*(1 - a1)).sum(axis=0)

        return (
            { 'bias_gradients': [grad_b1, grad_b2]
             , 'weight_gradients': [grad_w1, grad_w2]}
            , a2)

    def update_weights(self, learning_rate, gradients):
        self.layers[0].weight_array = self.layers[0].weight_array + learning_rate*gradients['weight_gradients'][0]
        self.layers[1].weight_array = self.layers[1].weight_array + learning_rate*gradients['weight_gradients'][1]

        self.layers[0].bias_vector = self.layers[0].bias_vector + learning_rate*gradients['bias_gradients'][0]
        self.layers[1].bias_vector = self.layers[1].bias_vector + learning_rate*gradients['bias_gradients'][1]


    def train(self, data, labels, learning_rate, iterations):
        for i in range(iterations):
            # get the gradients
            gradients, output = self.backward(data, labels)
            # use the gradients to updates the weights
            self.update_weights(learning_rate, gradients)
            c = cost(labels, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print("cost:", c, "classification_rate:", r)


class MyLayer:
    def __init__(self, weight_array, bias_vector, activation_function):
        self.weight_array = weight_array
        self.bias_vector = bias_vector
        self.activation_function = activation_function
