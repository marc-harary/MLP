import numpy as np

# http://neuralnetworksanddeeplearning.com/chap2.html

class NN:
    """ implements fully-connected feed-forward neural network """

    def __init__(self, ns, acts):
        # check for matching input dimensions
        assert(len(ns) >= 2)
        assert(all(map(lambda act: act in ["ReLU","sigmoid"], acts)))
        assert(len(ns) == len(acts))

        self.L = len(ns)
        self.deltas = self.dWs = self.dbs = self.L * [None]
        self.acts = acts
        self.bs = [np.zeros((n,1)) for n in ns[1:]] # 0 initialization
        self.Zs = (self.L-1) * [None]
        self.As = self.L * [None]
        self.Ws = [] # Xavier initialization for weight vectors
        for i in range(1,len(ns)):
            self.Ws.append( .01 * np.random.randn(ns[i-1],ns[i]) )
        

    def cross_entropy_loss(self, Y, lambd):
        """ returns cross-entropy loss with regularization """
        m = self.Ws[0].shape[1]
        A = self.As[-1]
        loss = -np.mean(Y*np.log(A) + (1-Y)*np.log(1-A)) # w/out reg
        regular = [np.square(W).sum() for W in self.Ws] # reg term
        regular = lambd/(2*m) * sum(regular)
        return loss + regular


    def evaluate(self, X, Y):
        """ evaluates F1 score of model given inputs and ground
            truths """
        pass


    def gPrime(self, act, z):
        """ implements derivatives of  various activation functions,
            specified by kwarg ACT """
        if act == "ReLU":
            return 0 if z < 0 else 1
        elif act == "sigmoid":
            return self.g("sigmoid",z) * (1-self.g("sigmoid",z))


    def g(self, act, z):
        """ implements various activation functions, specified by
            kwarg ACT """
        if act == "ReLU":
            return np.clip(z, a_min=0, a_max=None)
        elif act == "sigmoid":
            return (1+np.exp(-z))**-1


    def forward_propagate(self, X):
        """ propagates tensor forwards through network """
        assert(X.shape[0] == (self.Ws[0].shape[0]))
        self.As[0] = X
        for i in range(self.L-1):
            self.Zs[i] = self.Ws[i].T @ self.As[i] + self.bs[i]
            self.As[i+1] = self.g(self.acts[i], self.Zs[i])


    def back_propagate(self, Y):
        """ gets gradients for each layers given forward propagation
            has already occurred """
        # calculate delta for last layer
        dA = self.As[-1] - Y # dC/dAl
        self.deltas[-1] = dA * self.gPrime(self.acts[-1], self.Zs[-1])
        # calculate backprop for subsequent layers
        for l in range(self.L-2, 0, -1):
            self.deltas[l] = self.Ws[l] @ self.deltas[l+1]
            self.deltas[l] *= self.gPrime(self.acts[l], self.Zs[l])
            self.dWs[l] = self.As[l] @ self.deltas[l+1]
            self.dbs[l] = self.deltas[l+1]