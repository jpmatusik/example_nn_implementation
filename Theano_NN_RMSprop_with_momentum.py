from sklearn.utils import shuffle
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time

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

class HiddenLayer:
    def __init__(self, fan_in, fan_out, layer_id, activation_function=None):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.layer_id = layer_id
        # default to relu activation function
        self.activation_function = T.nnet.relu if activation_function is None else activation_function
        # initialize weights as standard normal with Var = 1/number of nodes in the layer
        W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0/fan_in)
        # initialize bias terms as 0
        B = np.zeros(fan_out)
        # create Theano shared variables and assign meaningful layer_ids for debugging
        self.W = theano.shared(W, 'W_%s' % self.layer_id)
        self.B = theano.shared(B, 'B_%s' % self.layer_id)

    def forward(self, X):
        # use alpha of .1 it activation function is relu
        if self.activation_function == T.nnet.relu:
            return self.activation_function(X.dot(self.W) + self.B, alpha=0.1)
        return self.activation_function(X.dot(self.W) + self.B)


# defaulting to the softmax output function
class TheanoNN:
    def __init__(self, input_layer_size, hidden_layer_sizes, output_layer_size, output_layer_act_f=T.nnet.softmax):
        self.input_layer_size = input_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size
        self.output_layer_act_f = output_layer_act_f

        # initialize NN layers - start with the hidden layers
        self.layers = []
        layer_id = 1
        # fan_in for first hidden layer is the number of features in the raw data
        fan_in = self.input_layer_size
        for fan_out in self.hidden_layer_sizes:
            self.layers.append(HiddenLayer(fan_in, fan_out, layer_id))
            # the fan_out of the current layer becomes the fan_in of the next
            fan_in = fan_out
            layer_id +=1

        # initialize layers - add the output layer
        self.layers.append(HiddenLayer(fan_in, self.output_layer_size, layer_id, self.output_layer_act_f))

        # collect all params in the model
        self.all_weights = []
        self.all_biases = []
        for layer in self.layers:
            self.all_weights += [layer.W]
            self.all_biases += [layer.B]

    def fit(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None, learning_rate=1e-3, mu=0.0, decay=0.999, l2_penalty=0, epochs=100, batch_size=None, print_period=100, show_fig=False):
        # in this case I'm assuming Ytrain is a vector of labels that go from 0 to K-1.

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
        # make sure to initialize param_deltas with 0's
        param_deltas = [theano.shared(np.zeros_like(p.get_value())) for p in (self.all_weights + self.all_biases)]
        # for RMS_prop
        # this can be initialized to 1's or 0's
        # better would be to use a bias correcting method.
        cache = [theano.shared(np.ones_like(p.get_value())) for p in (self.all_weights + self.all_biases)]

        # theano variables
        thX = T.matrix("thX")
        # this is a vector
        thY = T.lvector("thY")
        # This is a matrix
        Y_predicted_probs = self.forward(thX)

        # l2 regularization, scaled by the number of parameters
        l2_regularization = l2_penalty * T.sum([(p*p).sum() for p in self.all_weights]) / len(self.all_weights)
        # cross entropy, scaled by the number of observations
        # note that I'm indexing specifically the probs where Y is non-zero, as per the definition of cross entropy
        xent = -T.mean(T.log(Y_predicted_probs[T.arange(thY.shape[0]), thY]))
        cost = xent + l2_regularization
        prediction = self.predict(thX)

        # we have to update the following:
        # RMSprop cache
        # the param_deltas
        # the parameters themselves
        updates = (
          # RMSprop cache
          [(c, decay*c + (1-decay) * (T.grad(cost, p)**2)) for c, p in zip(cache, (self.all_weights + self.all_biases))]
            +
          # Parameters
          [(p, p + mu*dp - learning_rate * T.grad(cost, p)/T.sqrt(c + 1e-10) ) for p, c, dp in zip((self.all_weights + self.all_biases), cache, param_deltas)]
            +
          # Momentum update
          [(dp, mu*dp - learning_rate * T.grad(cost, p)/T.sqrt(c + 1e-10) ) for p, c, dp in zip((self.all_weights + self.all_biases), cache, param_deltas)]
        )

        # updates = [
        #     (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        # ] + [
        #     (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        # ]

        # feed all those updates into a the training function
        train_op = theano.function(
          inputs = [thX, thY]
        , updates = updates
        )

        # this function can used to report the cost and other performance metrics of the curent model
        cost_predict_op = theano.function(
          inputs=[thX, thY]
        , outputs=[cost, prediction]
        )

        if batch_size is None:
            batch_size = Xtrain.shape[0]

        n_batches = int(Xtrain.shape[0] / batch_size)
        self.costs = []
        for i in range(epochs):
            # reshuffle the data at the start of each epoch
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            for j in range(n_batches):

                start = time.time()
                # get batch of data
                Xbatch = Xtrain[j*batch_size:(j*batch_size+batch_size)]
                Ybatch = Ytrain[j*batch_size:(j*batch_size+batch_size)]

                # train the weights on this batch
                train_op(Xbatch, Ybatch)

                if j % print_period == 0:
                    c, p = cost_predict_op(Xbatch, Ybatch)
                    self.costs.append(c)
                    e = np.mean(Ybatch != p)
                    c = np.round(c, 4)
                    e = np.round(e, 4)
                    print("train: ", str(time.time() - start),  "i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

        if show_fig:
            plt.plot(self.costs)
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
