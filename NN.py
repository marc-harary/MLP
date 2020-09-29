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
        self.acts = acts
        self.deltas = self.L * [None]
        self.dWs    = self.L * [None]
        self.dbs    = self.L * [None]
        self.As     = self.L * [None]
        self.Zs     = self.L * [None]
        self.bs     = self.L * [None] # 0 initialization
        self.Ws     = self.L * [None] # Xavier initialization
        for i in range(1, self.L):
            self.Ws[i] = .01 * np.random.randn(ns[i-1], ns[i])
            self.bs[i] = np.zeros((ns[i],1))
        

    def cross_entropy_loss(self, Y, lambd):
        """ returns cross-entropy loss with regularization """
        m = self.Ws[1].shape[1]
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
            return (z >= 0).astype(np.float64)
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
        # check dimensions of input matrix match those of input layer
        assert(X.shape[0] == self.Ws[1].shape[0])

        # forward propagate and cache entire batch
        self.As[0] = X
        for i in range(1, self.L):
            self.Zs[i] = self.Ws[i].T @ self.As[i-1] + self.bs[i]
            self.As[i] = self.g(self.acts[i], self.Zs[i])

        # cache output for entire batch
        output = self.As[-1]

        # re-store average activations and weighted inputs
        self.As[0] = np.mean(X, axis=1, keepdims=True)
        for i in range(1, self.L):
            self.As[i] = np.mean(self.As[i], axis=1, keepdims=True)
            self.Zs[i] = np.mean(self.Zs[i], axis=1, keepdims=True)

        return output


    def back_propagate(self, Y):
        """ gets gradients for each layers given forward propagation
            has already occurred """
        # delta for last layer
        dA = np.mean(self.As[-1] - Y, axis=1, keepdims=True) 
        self.deltas[-1] = dA * self.gPrime(self.acts[-1], self.Zs[-1])
        self.dWs[-1] = self.deltas[-1] @ self.As[-2].T
        
        # deltas for preceding layers
        for l in range(self.L-2, 0, -1): 
            self.deltas[l] = self.Ws[l+1] @ self.deltas[l+1]
            self.deltas[l] *= self.gPrime(self.acts[l], self.Zs[l])
            self.dWs[l] = self.deltas[l] @ self.As[l-1].T
            self.dbs[l]=np.sum(self.deltas[l+1],axis=1,keepdims=True)

    
    def fit(self, X, Y, epochs=10_000, alpha=1e-2):
        for epoch in range(epochs):
            self.forward_propagate(X)
            self.back_propagate(Y)

            for l in range(1, self.L):
                self.Ws[l] -= alpha*self.dWs[l]
                self.bs[l] -= alpha*self.bs[l]

            loss = self.cross_entropy_loss(Y, lambd=0)
            if epoch % 1000 == 0:
                print("Loss for epoch {epoch} = {loss}"%(epoch,loss))