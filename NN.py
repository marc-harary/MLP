import numpy as np

class NN:
    """ implements fully-connected feed-forward neural network """


    def __init__(self, layers, activations):
        assert(len(layers) >= 2)
        assert(all(map(lambda x: x == "ReLU" or x == "sigmoid",
                       activations)))

        self.activations = activations
        self.bs = [np.zeros((layer,1)) for layer in layers]
        self.Ws = []
        for i in range(1,len(layers)):
            shape = (layers[i-1], layers[i])
            self.Ws.append(.01*np.random.randn(*shape))

    
    def XC(self, Y, lambd):
        """ returns cross-entropy loss """
        m = self.Ws[0].shape[1]
        A = self.As[-1]
        loss = -np.mean(Y*np.log(A) + (1-Y)*np.log(1-A))
        regular = [np.square(W).sum() for W in self.Ws]
        regular = lambd/(2*m) * sum(regular)
        return loss + regular


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
        self.As = []

        Z1 = self.Ws[0].T @ X
        if self.activations[0] == "ReLU":
            A1 = self.ReLU(Z1)
        else:
            A1 = self.sigmoid(Z1)
        self.Zs.append(Z1)
        self.As.append(A1)
        
        for i in range(1,len(self.Ws)):
            W = self.Ws[i]
            preA = self.As[i-1]
            curZ = W.T @ preA
            activation = self.activations[i]
            if activation == "ReLU":
                curA = self.ReLU(curZ)
            else:
                curA = self.sigmoid(curZ)
            self.Zs.append(curZ)
            self.As.append(curA)

    
    def back_propagate(self, Y):
        """ gets gradients for each layers given forward propagation
            has already occurred """
        m = self.Ws[0].shape[1]
        self.dZs = [self.As[-1] - Y]
        self.dWs = [self.dZs[0] @ self.As[-2].T]
        self.dbs = (1/m) * np.sum(self.dZs[0], axis=1, keepdims=True)

        for i in range(1, len(self.layers)):
            preW = self.Ws[-i]
            preZ = self.Zs[-i]
            preDZ = self.dZs[-i]
            curZ = self.Zs[-(i+1)]
            if self.activations[-(i+1)] == "ReLU":
                curDZ = preW.T @ preDZ * self.ReLUPrime(curZ)
            else:
                curDZ = preW.T @ preDZ * self.sigmoidPrime(curZ)
            curDw = (1/m)*dZ
            self.dZs.append(curDZ)