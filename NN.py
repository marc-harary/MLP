import numpy as np

# http://neuralnetworksanddeeplearning.com/chap2.html

class NN:
    """ implements fully-connected feed-forward neural network """

    def __init__(self, layers, activations):
        # check for matching input dimensions
        assert(len(layers) >= 2)
        assert(all(map(lambda x: x == "ReLU" or x == "sigmoid",
                       activations)))
        assert(len(layers) == len(activations))

        self.activations = activations
        self.L = len(layers)

        # zero initialization for bias vectors
        self.bs = [np.zeros((layer,1)) for layer in layers]
        # Xavier initialization for weight vectors
        self.Ws = []
        for i in range(1,len(layers)):
            shape = (layers[i-1], layers[i])
            self.Ws.append(.01*np.random.randn(*shape))

    
    def cross_entropy_loss(self, Y, lambd):
        """ returns cross-entropy loss with regularization """
        m = self.Ws[0].shape[1]
        A = self.As[-1]
        loss = -np.mean(Y*np.log(A) + (1-Y)*np.log(1-A))
        # calculate regularization term
        regular = [np.square(W).sum() for W in self.Ws]
        regular = lambd/(2*m) * sum(regular)
        return loss + regular


    def evaluate(self, X, Y):
        """ evaluates F1 score of model given inputs and ground
            truths """
        pass


    def ReLUPrime(self, z):
        """ derivative of ReLU activation function """
        return 0 if z < 0 else 1


    def sigmoidPrime(self, z):
        """ derivative of sigmoid activation function """
        return self.sigmoid(z)*(1-self.sigmoid(z))


    def sigmoid(self, z):
        """ sigmoid activation function """
        return (1+np.exp(-z))**-1


    def ReLU(self, z):
        """ ReLU activation function """
        return np.clip(z, a_min=0, a_max=None)


    def forward_propagate(self, X):
        """ propagates tensor forwards through network """
        assert(X.shape[0] == (self.Ws[0].shape[0]))
        self.Zs = []
        self.As = [X]

        # calculate activation for first layer
        Z1 = self.Ws[0].T @ X
        if self.activations[0] == "ReLU":
            A1 = self.ReLU(Z1)
        else:
            A1 = self.sigmoid(Z1)
        self.Zs.append(Z1)
        self.As.append(A1)
        
        # calculate activations for subsequent layers
        for i in range(1, len(self.Ws)):
            Z = self.Ws[i].T @ self.As[i] + self.bs[i]
            if self.activations[i] == "ReLU":
                A = self.ReLU(Z)
            else:
                A = self.sigmoid(Z)
            self.Zs.append(Z)
            self.As.append(A)

    
    def back_propagate(self, Y):
        """ gets gradients for each layers given forward propagation
            has already occurred """
        m = self.Ws[0].shape[1]
        
        self.deltas = self.dAs = self.dWs = self.dbs = self.L * [None]
    
        # calculate backprop for last layer
        self.dAs[self.L-1] = self.As[self.L-1] - Y
        if self.activations[self.L-1] == "ReLU":
            # apply proper g'(.)
            self.deltas[self.L-1] = self.dAs[self.L-1] *\
                self.ReLUPrime(self.Zs[self.L-1])
        else:
            self.deltas[self.L-1] = self.dAs[self.L-1] *\
                self.sigmoidPrime(self.Zs[self.L-1])

        # calculate backprop for subsequent layers
        for l in range(self.L-2, 1, -1):
            sigmaPrime = self.ReLUPrime if self.activations[l] ==\
                "ReLU" else self.sigmoidPrime
            self.deltas[l] = (self.Ws[l+1].T @ self.deltas[l+1]) *\
                sigmaPrime(self.Zs[l])