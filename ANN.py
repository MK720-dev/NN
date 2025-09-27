from turtle import forward
from numpy import array, zeros, ones, arange, exp, dot, save, pi, linspace,\
                  matrix, ceil, mean, meshgrid, stack, mod
from numpy.random import randn, randint, uniform, normal, seed, shuffle
from numpy.linalg import norm
import matplotlib.pylab as plt
import sys
import numpy as np
from matplotlib import rc
from optimizers import BatchGD, LineSearchGD, BFGS_Wolfe, ScipyBFGS, LineSearchSGD, MiniBatchBFGS_Wolfe

rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern']})
rc('text', usetex = False)

class GeneralNetwork:
    ''' A GeneralNetwork object contains information about network architecture.
        These include: weights, biases, number of layers, etc. Current
        implementation focuses solely on fully connected networks, though we
        plan to incorporate other variants.

    '''
    def __init__(self, number_of_layers : int, neurons_per_layer : list, verbose = 0, \
                       activation_function = "sigmoid", vis = False):
        '''
        Inputs: number_of_layers  (int)  : specifies number of layers in the network.
                neurons_per_layer (list) : specifies number of neurons in the network.
                                          Each entry gives the total number of neurons
                                          for one layer.
                verbose       (boolean) : output flag for user.
        Outputs: N/A
        Description: Initialization function of a GeneralNetwork object.
        '''

        try:
            if number_of_layers != len(neurons_per_layer):
                print("The length of the neuron vector has to match the number \
                        of layers. Each entry refers to the number of neurons in\
                        that layer!")
                sys.exit()
        except:
            print("The variable number_of_layers must be an integer!")
            print("The variable neurons_per_layer must be a list!")
            sys.exit()

        list_of_weight_matrices = [normal(size=(neurons_per_layer[0], \
                                                neurons_per_layer[0]))]
        # create first entry connecting input layer to first hidden layer
        list_of_bias_vectors    = [normal(size=(neurons_per_layer[0], 1))]
        # Generate first entry in the bias vector. It corresponds to the bias
        # for the first layer


        # Initialization
        self.weights             = list_of_weight_matrices
        self.biases              = list_of_bias_vectors

        self.number_of_layers    = number_of_layers
        self.neurons_per_layer   = neurons_per_layer

        self.verbose             = verbose

        self.activation          = []
        self.delta               = []

        self.activation_function = activation_function
        self.vis                 = vis

        # Populate list of weight matrices per layer
        for i in arange(1, len(neurons_per_layer)):
            self.weights.append(normal(size=(neurons_per_layer[i],\
                                        neurons_per_layer[i - 1])))
            self.biases.append(normal(size=(neurons_per_layer[i], 1)))

    def forward(self, x):
        """
        Run forward pass and return (list of activations, list of pre-activations).
        """
        activations, zs = [], []
        a = x
        for s in range(self.number_of_layers):
            z = self.weights[s] @ a + self.biases[s]
            a = self.activate(a, self.weights[s], self.biases[s])
            zs.append(z)
            activations.append(a)
        return activations, zs

    def activate(self, x : array, W : array, b : array):
        '''
        Inputs:    x (1D numpy array) : input from previous layer
                   W (2D numpy array) : weight matrix for current layer
                   b (1D numpy array) : bias vector   for current layer
        Outputs: Sigmoid function output, this is the activation function that
                 is currently applied (between 0 and 1).

        Description: Function calculates activation for a layer, where each
                     entry in the activation array represents the activation for
                     one neuron.
        '''

        if self.activation_function == "sigmoid":
            return 1 / (1 + exp(-(dot(W, x) + b)))
        elif self.activation_function == "relu":
            dummy = dot(W, x) + b
            for i in arange(dummy.shape[0]):
                if dummy[i, 0] > 0:
                    pass
                else:
                    dummy[i, 0] = 0.0
            return dummy
        elif self.activation_function == "leakyrelu":
            dummy = dot(W, x) + b
            for i in arange(dummy.shape[0]):
                if dummy[i, 0] > 0:
                    pass
                else:
                    dummy[i, 0] = 0.01 * dummy[i, 0]
            return dummy

    def predict(self, x : array):
        for s in arange(self.number_of_layers):
            a = self.activate(x, self.weights[s], self.biases[s])
            x = a

        return a

    def gradient(self, x : array):
        '''

        Inputs      : x     (2D numpy array)
        Outputs     : dummy (2D numpy array)
        Description : Computes gradient for each type of activation function

        Some errors still exist here.
        '''
        if self.activation_function == "sigmoid":
            return x * (1 - x)
        elif self.activation_function == "relu":
            dummy = x
            for i in arange(x.shape[0]):
                if dummy[i, 0] > 0:
                    dummy[i, 0] = 1.0

                else:
                    dummy[i, 0] = 0.0
            return dummy
        elif self.activation_function == "leakyrelu":
            dummy = x
            for i in arange(x.shape[0]):
                if dummy[i, 0] > 0:
                    dummy[i, 0] = 1.0

                else:
                    dummy[i, 0] = 0.01 * dummy[i, 0]
            return dummy

        return

    def train(self, Data, eta : float = 1, epochs : int = 10000, replacement : bool = 0):
        '''
        Inputs:   Data        (object) : this object contains information
                                         about the training data. The object
                                         has information about LABELLED data
                                         (x_train, y_train) pairs.
                  eta         (float)  : learning rate, i.e. how much we update.
                                         This is currently constant.
                  epochs      (int)    : number of cycles over a complete update
                                         of training data.
                  replacement (int)    : variable decides whether to use each
                                         piece of training data once in an epoch
                                         or sometimes some multiple and others
                                         not at all.
        Outputs:  cost (2D numpy array): contains the cost measurement after each
                                         update, per epoch.

        Description: This function performs gradient descent with respect to the
                     training data. It finds weights and biases which are tuned
                     such that the f_NN(x_train) = y_train, where f_NN is a function
                     outlining the architecture of the neural network.
        '''
        cost   = zeros((epochs, Data.xtrain.shape[1]))
        # create empty 2D numpy array for storage
        xtrain = zeros((Data.xtrain.shape[0], 1))
        # create empty 1D numpy array which will be overwritten with training data
        ytrain = zeros((Data.ytrain.shape[0], 1))
        # create empty 1D numpy array which will be overwritten with training data
        for update in arange(epochs):
            # begin looping over training data epochs many times
            shuffled_ints = array(arange(Data.xtrain.shape[1]))
            # generate integer array to access specific training data per update
            shuffle(shuffled_ints)
            # shuffle integers for no bias toward certain configuration
            for counter in arange(shuffled_ints.shape[0]):
                # begin cycle over training set
                if not replacement:
                        k = shuffled_ints[counter]
                        # without replacement
                else:
                        k = randint(Data.xtrain.shape[1])
                        # with replacement

                # Training Data
                xtrain[:, 0], ytrain[:, 0] = Data.xtrain[:, k], Data.ytrain[:, k]

                # Forward Prop
                for s in arange(self.number_of_layers):
                    self.activation.append(self.activate(xtrain,            \
                        self.weights[s], self.biases[s]))
                    xtrain = self.activation[s]

                #  Back Prop
                self.delta.append(self.gradient(self.activation[-1]) * (self.activation[-1] - ytrain))
                for s in arange(0, self.number_of_layers-1):
                    self.delta.append(self.gradient(self.activation[-2 - s]) * \
                    dot(self.weights[-1  - s].T, self.delta[s]))

                #  Update weights
                self.weights[0]  -= eta * self.delta[-1] * Data.xtrain[:, k].T
                for s in arange(1, self.number_of_layers):
                    if s >= 1:
                        self.weights[s]   -= eta * dot(self.delta[-(s + 1)], self.activation[s - 1].T)

                deltaflip = self.delta
                deltaflip.reverse()
                # reverse to put bias update in loop


                # Update biases
                for s in arange(self.number_of_layers):
                    self.biases[s] -= eta * deltaflip[s]

                # Reset activation and errors for next loop
                self.activation = []
                self.delta      = []

                # Save cost
                cost[update, counter] = self.cost_function(Data)
            if update == epochs - 1 and self.vis == True:
               self.visual(Data)
            if self.verbose:
                # verbosity flag prints to console
                print('Average cost for epoch ', update + 1, 'is :', mean(cost[update, :]))

        return cost

    def visual(self, Data, no_of_points : int = 1000):
        '''
        '''
        x    = zeros((2,1))

        X, Y = meshgrid(linspace(0, 1, no_of_points), linspace(0, 1, no_of_points))
        # create grid of points
        X1, X2 = array(X.ravel()), array(Y.ravel())
        # vectorize
        Stack  = stack((X1, X2), axis = 1)
        # left to right stack, columnwise
        empty  = zeros(X.shape[0]*X.shape[0])
        # matrix of values

        for i in arange(Stack.shape[0]):
            # run over each coordinate
            x[0,0], x[1,0] = Stack[i, 0], Stack[i, 1]
            # input
            Predictions = self.predict(x)
            # predict
            Predictions = array(Predictions[0] >= Predictions[1])
            # create booleans
            if Predictions[0] == True:
                empty[i] = 1
                # assign 1 where true

        Pred = empty.reshape((X.shape[0], X.shape[0]))
        # reshape ready for plotting contour
        import matplotlib.pyplot as plt
        plt.style.use('Solarize_Light2')
        plt.figure()
        # plot figure
        plt.contourf(X, Y, Pred, cmap = 'cividis', alpha = 0.8)
        # plot contour

        plt.scatter(Data.xtrain[0, 0:5], Data.xtrain[1, 0:5], marker='^', c = 'k', lw=3)
        # plot first half of training set
        plt.scatter(Data.xtrain[0, 5:], Data.xtrain[1, 5:], marker='o', c='w', lw=3)
        # plot second half of training set
        if Data.highamdata:
            plt.savefig('higham_planes.png')
            # save figure
        else:
            plt.savefig('planes.png')
            # save figure
        plt.show()
        return
    
    def cost_function(self, Data):
        '''
        Inputs      : Data (object) - contains information on all training
                                      data.

        Outputs     : evaluated cost function.

        Description : This function evaluates the accuracy of the trained
                      function with the actual data.

        '''
        temp_cost = zeros((Data.xtrain.shape[1],1))
        # generate zeros 2D numpy array for storing cost
        x         = zeros((Data.xtrain.shape[0],1))
        # generate zeros 2D numpy array for storing training data before
        # evaluation
        for i in arange(temp_cost.shape[0]):
            # begin loop for computing each individual cost from each piece
            # of training data
            x[0,0], x[1,0] = Data.xtrain[0,i], Data.xtrain[1,i]

            # Feedforward
            for s in arange(self.number_of_layers):
                a = self.activate(x, self.weights[s], self.biases[s])
                x = a

            temp_cost[i] = norm(x.ravel() - Data.ytrain[:, i], 2)

        return norm(temp_cost, 2)**2
    #--------------------------------------------------------
    # NEW CODE: Extending the Newtwork's Base Class to
    # handle the new full batch optimizers 
    # (LineSearchGD, BFGS,...)
    #--------------------------------------------------------

    # ---- Helper: remember shapes for packing/unpacking ----
    def _param_shapes(self):
        w_shapes = [w.shape for w in self.weights]
        b_shapes = [b.shape for b in self.biases]
        return w_shapes, b_shapes 

    #---- Helper: Flattens weights matrix and biases vector to be used for batch cost and gradient function computations ----
    def pack_params(self):
        flat = []
        for w in self.weights: flat.append(w.ravel())
        for b in self.biases: flat.append(b.ravel())
        return np.concatenate(flat)

    #---- Helper: Puts Weights and Biases Back into their original form ----
    def unpack_params(self, theta):
        w_shapes, b_shapes = self._param_shapes()
        ofs = 0
        for i, sh in enumerate(w_shapes):
            sz = sh[0]*sh[1]
            self.weights[i] = theta[ofs:ofs+sz].reshape(sh)
            ofs += sz

        for i, sh in enumerate(b_shapes):
            sz = sh[0]*sh[1]
            self.biases[i] = theta[ofs:ofs+sz].reshape(sh)
            ofs += sz

    #---- Helper: Cost as function of theta ----#
    def cost_from_theta(self, theta, Data):
        self.unpack_params(theta)
        return self.cost_function(Data)

    #---- Helper: Gradient wrt theta (full batch) ----#
    def grad_from_theta(self, theta, Data):
        self.unpack_params(theta)
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        x = np.zeros((Data.xtrain.shape[0], 1))
        y = np.zeros((Data.ytrain.shape[0],1))

        N =  Data.xtrain.shape[1]

        for k in range(N):
            # sample 
            x[:, 0], y[:,0] = Data.xtrain[:,k], Data.ytrain[:,k]

            # forward 
            acts = []
            a = x
            for s in range(self.number_of_layers):
                a = self.activate(a, self.weights[s], self.biases[s])
                acts.append(a)

            # backprop
            deltas = []
            deltas.append(self.gradient(acts[-1]) * (acts[-1] - y))
            for s in range(0, self.number_of_layers-1):
                deltas.append(self.gradient(acts[-2 - s]) * (self.weights[-1 - s].T @ deltas[s]))

            # grads
            dW[0] += deltas[-1] @ x.T
            for s in range(1, self.number_of_layers):
                dW[s] += deltas[-(s+1)] @ acts[s-1].T
            deltas.reverse()
            for s in range(self.number_of_layers):
                db[s] += deltas[s]

        flats = [g.ravel() for g in dW] + [g.ravel() for g in db]
        return np.concatenate(flats)

    #---- Helper ----
    def loss_single(self, theta, x, y):
        """Compute loss for a single sample."""
        self.unpack_params(theta)
        a = x
        for s in range(self.number_of_layers):
            a = self.activate(a, self.weights[s], self.biases[s])
        return 0.5 * np.linalg.norm(a - y)**2

    #---- Helper ----
    def grad_single(self, theta, x, y):
        """Compute gradient wrt theta for a single sample (backprop)."""
        self.unpack_params(theta)
        acts = []
        a = x
        for s in range(self.number_of_layers):
            a = self.activate(a, self.weights[s], self.biases[s])
            acts.append(a)

        deltas = [self.gradient(acts[-1]) * (acts[-1] - y)]
        for s in range(0, self.number_of_layers-1):
            deltas.append(self.gradient(acts[-2-s]) *
                          (self.weights[-1-s].T @ deltas[s]))

        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        dW[0] = deltas[-1] @ x.T
        for s in range(1, self.number_of_layers):
            dW[s] = deltas[-(s+1)] @ acts[s-1].T

        deltas.reverse()
        for s in range(self.number_of_layers):
            db[s] = deltas[s]

        return np.concatenate([w.ravel() for w in dW] +
                              [b.ravel() for b in db])

    #---- Helper ----
    def loss_batch(self, theta, X_batch, Y_batch):
        self.unpack_params(theta)
        m = X_batch.shape[1]
        loss = 0.0
        for i in range(m):
            a = X_batch[:, [i]]
            y = Y_batch[:, [i]]
            for s in range(self.number_of_layers):
                a = self.activate(a, self.weights[s], self.biases[s])
            loss += 0.5 * np.linalg.norm(a - y)**2
        return loss / m

    #---- Helper ----
    def grad_batch(self, theta, X_batch, Y_batch):
        self.unpack_params(theta)
        m = X_batch.shape[1]
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        for i in range(m):
            x = X_batch[:, [i]]
            y = Y_batch[:, [i]]

            # forward
            acts = []
            a = x
            for s in range(self.number_of_layers):
                a = self.activate(a, self.weights[s], self.biases[s])
                acts.append(a)

            # backprop
            deltas = [self.gradient(acts[-1]) * (acts[-1] - y)]
            for s in range(0, self.number_of_layers-1):
                deltas.append(self.gradient(acts[-2-s]) *
                              (self.weights[-1-s].T @ deltas[s]))

            # accumulate grads
            dW[0] += deltas[-1] @ x.T
            for s in range(1, self.number_of_layers):
                dW[s] += deltas[-(s+1)] @ acts[s-1].T
            deltas.reverse()
            for s in range(self.number_of_layers):
                db[s] += deltas[s]

        # average
        dW = [dw / m for dw in dW]
        db = [dbi / m for dbi in db]
        return np.concatenate([w.ravel() for w in dW] +
                              [b.ravel() for b in db])



    #---- NEW Training Function: Trains Neural Network using specific optimizer ----
    def train_with_optimizer(self, Data, optimizer, steps=1000, verbose=True):
        theta = self.pack_params()
        grad = self.grad_from_theta(theta, Data)

        history = []
        for t in range(steps):
            cost = self.cost_from_theta(theta, Data)
            history.append(cost)

            if isinstance(optimizer, BatchGD):
                theta = optimizer.step(theta, grad)
                grad = self.grad_from_theta(theta, Data)
            elif isinstance(optimizer, LineSearchGD):
                theta = optimizer.step(theta, grad,
                                       lambda th: self.cost_from_theta(th, Data),
                                       lambda th: self.grad_from_theta(th, Data))
                grad = self.grad_from_theta(theta, Data)
            elif isinstance(optimizer, BFGS_Wolfe):
                theta, grad = optimizer.step(theta, grad,
                                            lambda th: self.cost_from_theta(th, Data),
                                            lambda th: self.grad_from_theta(th, Data))
            elif isinstance(optimizer, ScipyBFGS):
                theta, grad = optimizer.step(theta, grad,
                                            lambda th: self.cost_from_theta(th, Data),
                                            lambda th: self.grad_from_theta(th, Data))
            else:
                raise ValueError("Unknown optimizer type")

            if verbose and (t % max(1, steps//10) == 0):
                print(f"Step {t}, cost={cost:.6f}")

        self.unpack_params(theta)
        return np.array(history)

    #---- NEW Training Function: Trains Neural Network using specific Line Search SGD (Armijo)----
    def train_with_linesearch_sgd(self, Data, optimizer, epochs=50, verbose=True):
        theta = self.pack_params()
        history = []

        for ep in range(epochs):
            indices = np.random.permutation(Data.xtrain.shape[1])
            for k in indices:
                x = Data.xtrain[:, [k]]
                y = Data.ytrain[:, [k]]

                grad = self.grad_single(theta, x, y)

                theta = optimizer.step(
                                        theta, grad,
                                        lambda th, x, y: self.loss_single(th, x, y),
                                        lambda th, x, y: self.grad_single(th, x, y),
                                        x, y
                                    )

            cost = self.cost_from_theta(theta, Data)
            history.append(cost)
            if verbose:
                print(f"Epoch {ep+1}, cost={cost:.6f}")

        self.unpack_params(theta)
        return np.array(history)

    #---- NEW Training Function: Trains Neural Network using stochastic BFGS----
    def train_with_minibatch_bfgs(self, Data, optimizer, epochs=50, verbose=True):
        theta = self.pack_params()
        history = []

        for ep in range(epochs):
            indices = np.random.permutation(Data.xtrain.shape[1])
            for i in range(0, len(indices), optimizer.batch_size):
                batch_idx = indices[i:i+optimizer.batch_size]
                X_batch = Data.xtrain[:, batch_idx]
                Y_batch = Data.ytrain[:, batch_idx]

                grad = self.grad_batch(theta, X_batch, Y_batch)

                theta, grad = optimizer.step(
                    theta, grad,
                    lambda th, Xb, Yb: self.loss_batch(th, Xb, Yb),
                    lambda th, Xb, Yb: self.grad_batch(th, Xb, Yb),
                    X_batch, Y_batch
                )

            cost = self.cost_from_theta(theta, Data)
            history.append(cost)
            if verbose:
                print(f"Epoch {ep+1}, cost={cost:.6f}")

        self.unpack_params(theta)
        return np.array(history)


    #---- Helper: New Generic Visualization function that works for any number of output classes ----
    def plot_decision_boundary(self, Data, title="", ax=None): 
        xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        grid = np.vstack([xx.ravel(), yy.ravel()])

        preds = []
        for j in range(grid.shape[1]):
            out = self.predict(grid[:, [j]])
            preds.append(np.argmax(out))   # class index
        preds = np.array(preds).reshape(xx.shape)

        if ax is None:
            ax = plt.gca()
        ax.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.coolwarm)
        ax.scatter(Data.xtrain[0, :], Data.xtrain[1, :],
                   c=np.argmax(Data.ytrain, axis=0),
                   cmap=plt.cm.coolwarm, edgecolors="k")
        ax.set_title(title)


        
class Data:
    ''' Data object contains information about training data. Its initialization
        generates the training data for this particular problem
    '''
    def __init__(self, number_of_data_points : int, highamdata = True):
        '''
        Inputs: number_of_data_points (int)     - total data points without higham
                                                  data.
                highamdata            (boolean) - whether to use data from higham
                                                  paper or not.
        '''

        self.highamdata = highamdata

        if not self.highamdata:
            # generate data
            self.x           = linspace(0, 1, number_of_data_points)
            x1               = zeros((1, number_of_data_points))
            x2               = zeros((1, number_of_data_points))
            x1[0,:]          = uniform(0, 1, self.x.shape[0])
            x2[0,:]          = uniform(0, 1, self.x.shape[0])
            y1               = zeros((1, int(number_of_data_points / 2)))
            # assign labelling
            y2               = ones((1,  int(number_of_data_points / 2)))
            # assign labelling
            #y3               = zeros((1, int(number_of_data_points / 2)))
        else:
            # use higham data
            x1 = array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
            x2 = array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])


            y1               = zeros((1, int(number_of_data_points / 2)))
            # assign labelling
            y2               = ones((1,  int(number_of_data_points / 2)))
            # assign labelling

            # zeros arrays for training data

        self.xtrain      = zeros((2, number_of_data_points))
        self.ytrain      = zeros((2, number_of_data_points))

            # begin assignment
        self.xtrain[0,:], self.xtrain[1,:]                = x1, x2
        self.ytrain[0, 0:int(number_of_data_points / 2)]  = y2
        self.ytrain[0, int(number_of_data_points   / 2):] = y1
        self.ytrain[1, 0:int(number_of_data_points / 2)]  = y1
        self.ytrain[1, int(number_of_data_points   / 2):] = y2


if __name__ == "__main__":

    # Data: Higham 10-point toy dataset
    data = Data(number_of_data_points=10, highamdata=True)

    def fresh_net():
        return GeneralNetwork(number_of_layers=3,
                              neurons_per_layer=[2, 5, 2],
                              activation_function="sigmoid")
    
    # ----------------------------
    # 1) Original training (per-sample SGD)
    # ----------------------------
    net_orig = fresh_net()
    cost_grid = net_orig.train(data, epochs=10000)
    hist_orig = cost_grid.mean(axis=1)  # average per epoch

    # ----------------------------
    # 2) Batch Gradient Descent
    # ----------------------------
    net_batch = fresh_net()
    hist_batch = net_batch.train_with_optimizer(data, BatchGD(lr=0.1),
                                                steps=10000, verbose=False)

    # ----------------------------------------
    # 3) Armijo Line Search Gradient Descent
    # ----------------------------------------
    net_ls = fresh_net()
    hist_ls = net_ls.train_with_optimizer(data, LineSearchGD(),
                                          steps=10000, verbose=False)
    # --------------------------------
    # 4) BFGS + Weak Wolfe Line Search
    # --------------------------------
    net_bfgs = fresh_net()
    hist_bfgs = net_bfgs.train_with_optimizer(data, BFGS_Wolfe(), 
                                              steps=5000, verbose=False)

    # --------------------------------
    # 5) Scipy's BFGS
    # --------------------------------
    net_scipy_bfgs = fresh_net()
    hist_scipy_bfgs = net_scipy_bfgs.train_with_optimizer(data, ScipyBFGS(), steps=2000, verbose=False)

    # --------------------------------
    # 6) Linea Search SGD (Armijo)
    # --------------------------------
    """net_sgd_ls = fresh_net()
    hist_sgd_ls = net_sgd_ls.train_with_linesearch_sgd(data, LineSearchSGD(), epochs=10000)"""

    # --------------------------------
    # 7) Stochastic BFGS
    # --------------------------------
    """net_sbfgs = fresh_net()
    hist_sbfgs = net_sbfgs.train_with_minibatch_bfgs(data, MiniBatchBFGS_Wolfe(), epochs=10000)"""

    # ----------------------------
    # Plot cost comparison
    # ----------------------------
    plt.figure()
    plt.plot(hist_orig, label="Original SGD (per-sample)")
    plt.plot(hist_batch, label="BatchGD (fixed η)")
    plt.plot(hist_ls, label="LineSearchGD (Armijo)")
    plt.plot(hist_bfgs, label="BFGS_Wolfe")
    plt.plot(hist_scipy_bfgs, label="Scipy_BFGS")
    #plt.plot(hist_sgd_ls, label="SGD + Armijo Line Search")
    #plt.plot(hist_sbfgs, label="Stochastic BFGS")
    plt.yscale("log")
    plt.xlabel("Step / Epoch")
    plt.ylabel("Cost")
    plt.title("Optimizer Comparison on Higham Data")
    plt.legend()
    plt.show()

    # -------------------------------
    # Plot classification boundaries
    # -------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    nets = [
        (net_orig, "Original SGD"),
        (net_batch, "BatchGD"),
        (net_ls, "LineSearchGD"),
        (net_bfgs, "BFGS_Wolfe"),
        (net_scipy_bfgs, "Scipy_BFGS")
        #(net_sgd_ls, "SGD + Armijo Line Search"),
        #(net_sbfgs, "Stochastic BFGS")
    ]

    for ax, (net, title) in zip(axes.ravel(), nets):
        net.plot_decision_boundary(data, title=title, ax=ax)

    plt.tight_layout()
    plt.show()