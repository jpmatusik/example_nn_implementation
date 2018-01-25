from sklearn.utils import shuffle
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

# to define a hidden layer we need:
# - fan_in
#   - used to initialize weights
# - fan_out
#   - used to initialize weights
#   - used to initialize biad
# - activation function
# - a forward method to calculate the activations

# Because we're using Theano we need the following:
# - the W matrix and B vector must be shared variables
#   - this lets us not have to pass W and B as inputs into theano functions
#   - this les us take advantage of the 'updates' argument of theano functions
#     - this is key to conveniently updating the weights and bias terms using the gradient
# - layer_id to uniquely identify the W and B between layers.
# - the last thing we need is self.params. It's not obvious why we'd need this here until
#   we go to calculate things. It just makes things a little nicer.

class HiddenLayer:
    def __init__(self, fan_in, fan_out, layer_id, activation_function=None):
        self.layer_id = layer_id
        # initialize weights as standard normal with Var = 1/number of nodes in the layer
        W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_out)
        # initialize bias terms as 0
        B = np.zeros(fan_out)
        # create Theano shared variables and assign meaningful layer_ids for debugging
        self.W = theano.shared(W, 'W_%s' % self.layer_id)
        self.B = theano.shared(B, 'B_%s' % self.layer_id)
        self.params = [self.W, self.B]
        # default to relu activation function
        self.activation_function = T.nnet.relu if activation_function is None else activation_function

    def forward(self, X):
        return self.activation_function(X.dot(self.W) + self.B)

class TheanoNN:
    def __init__(self, input_layer_size, hidden_layer_sizes, output_layer_size, output_layer_act_f=T.nnet.softmax):
        self.input_layer_size = input_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size
        self.output_layer_act_f = output_layer_act_f
        self.layers = []
        self.params = []

        # initialize layers - starting with the hidden layers
        layer_id = 1
        fan_in = self.input_layer_size
        for fan_out in self.hidden_layer_sizes:
            self.layers.append(HiddenLayer(fan_in, fan_out, layer_id))
            layer_id +=1
            # the fan_out of the current layer becomes the fan_in of the next
            fan_in = fan_out
        # initialize layers - add the output layer
        self.layers.append(HiddenLayer(fan_in, self.output_layer_size, layer_id, self.output_layer_act_f))

        # collect all params in the model
        for layer in self.layers:
            self.params += layer.params

    def fit(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None, learning_rate=1e-3, mu=0.99, decay=0.999, epochs=500, batch_size=50, l2_penalty=1e-4, show_fig=False):
        # mu is for momentum
        # decay is for RMSprop
        #   - i.e. the parameter to calculate the EWMA for each of the caches

        # if no validation data is given, we make our own.
        if Xvalid is None:
            # random shuffle before we split the data
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            # calculate the number of training examples
            n_train = int(.7*Xtrain.shape[0])
            # split the data
            Xvalid = Xtrain[n_train:,:]
            Yvalid = Ytrain[n_train:]
            Xtrain = Xtrain[:n_train,:]
            Ytrain = Ytrain[:n_train]

        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        # for RMS_prop
        cache = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        # theano variables
        thX = T.matrix("thX")
        thY_actual_labels = T.lvector("thY_actual_labels")
        # This is a matrix
        Y_predicted_probs = self.forward(thX)

        # inputs for cost function
        # TODO: in this case we're performing regularization on the bias terms as well... that doesn't make sense
        # Solution would be to split up the weights and biases into their own instance variables.
        l2_regularization = T.sum([(p * p).sum() for p in self.params])
        # note that I'm indexing specifically the probs where Y is non-zero, as per the definition of cross entropy
        xent = -T.log(Y_predicted_probs[T.arange(thY_actual_labels.shape[0]), thY_actual_labels])
        cost = T.mean(xent) + l2_penalty*l2_regularization
        prediction = self.predict(thX)

        cost_predict_op = theano.function(
          inputs=[thX, thY_actual_labels]
        , outputs=[cost, prediction]
        )

        updates = (
        # cache updates for RMS_prop
          [(c, decay*c + (1-decay)*(T.grad(cost, p)**2)) for c, p in zip(cache, self.params)]
            +
          [(p, p + mu*dp  - learning_rate * T.grad(cost, p)/T.sqrt(c + 1e-10)) for p, c, dp in zip(self.params, cache, dparams)]
            +
          [(dp, mu*dp - learning_rate * T.grad(cost, p)/T.sqrt(c + 1e-10)) for p, c, dp in zip(self.params, cache, dparams)]
        )

        train_op = theano.function(
          inputs = [thX, thY_actual_labels]
        , updates = updates
        )

        n_batches = Xtrain.shape[0] // batch_size
        costs = []
        for i in range(epochs):
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_size:(j*batch_size+batch_size)]
                Ybatch = Ytrain[j*batch_size:(j*batch_size+batch_size)]

                train_op(Xbatch, Ybatch)

                if j % 20 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        # initial output is the input data itself
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X):
        Y_predicted_probs = self.forward(X)
        return T.argmax(Y_predicted_probs, axis=1)



X, Y = getData()
# X, Y = getBinaryData()
model = TheanoNN(input_layer_size=2304, hidden_layer_sizes=[2000,1000], output_layer_size=7)
model.fit(X, Y, learning_rate=1e-10, mu=0.99, decay=0.999, epochs=10, batch_size=1006, l2_penalty=1e-4, show_fig=True)
