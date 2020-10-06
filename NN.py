import numpy as np

# http://neuralnetworksanddeeplearning.com/chap2.html

class NN:
    """ implements fully-connected feed-forward neural network """

    def __init__(self, ns, acts):
        # check for matching input dimensions
        assert(len(ns) >= 2)
        assert(all(map(lambda act: act in ["ReLU","sigmoid"], acts)))
        assert(len(ns) == len(acts))

        self.ns     = ns
        self.L      = len(ns)
        self.acts   = acts
        self.deltas = self.L * [None]
        self.dWs    = self.L * [None]
        self.dbs    = self.L * [None]
        self.As     = self.L * [None]
        self.Zs     = self.L * [None]
        self.bs     = self.L * [None] # 0 initialization
        self.Ws     = self.L * [None] # Xavier initialization
        self.output = None
        for i in range(1, self.L):
            self.Ws[i] = .01 * np.random.randn(ns[i], ns[i-1])
            self.bs[i] = np.zeros((ns[i],1))
        

    def cross_entropy_loss(self, X, Y, lambd):
        """ returns cross-entropy loss with regularization """
        m = self.Ws[1].shape[1]
        A = self.forward_propagate(X)
        loss = -np.mean(Y*np.log(A) + (1-Y)*np.log(1-A)) # w/out reg
        regular = [np.square(W).sum() for W in self.Ws
                   if W is not None]
        regular = lambd/(2*m) * sum(regular)
        return loss + regular


    def evaluate(self, X, Y):
        """ evaluates F1 score of model given inputs and ground
            truths """
        A = self.forward_propagate(X) > .5
        return np.mean(A == Y)


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
            return 1/(1+np.exp(-z))


    def forward_propagate(self, X):
        """ propagates tensor forwards through network """
        # check dimensions of input matrix match those of input layer
        assert(X.shape[0] == self.Ws[1].shape[1])

        # forward propagate and cache entire batch
        self.As[0] = X
        for i in range(1, self.L):
            self.Zs[i] = self.Ws[i] @ self.As[i-1] + self.bs[i]
            self.As[i] = self.g(self.acts[i], self.Zs[i])

        # cache output for entire batch
        self.output = self.As[-1]

        # re-store average activations and weighted inputs
        self.As[0] = np.mean(X, axis=1, keepdims=True)
        for i in range(1, self.L):
            self.As[i] = np.mean(self.As[i], axis=1, keepdims=True)
            self.Zs[i] = np.mean(self.Zs[i], axis=1, keepdims=True)

        return self.output


    def back_propagate(self, Y):
        """ gets gradients for each layers given forward propagation
            has already occurred """
        # delta for last layer
        self.deltas[-1] = np.mean(self.output-Y, axis=1,
                                  keepdims=True) *\
                            self.gPrime(self.acts[-1], self.Zs[-1])

        # deltas for preceding layers
        for l in range(self.L-2, 0, -1):
            self.deltas[l] = self.Ws[l+1].T @ self.deltas[l+1] *\
                self.gPrime(self.acts[l], self.Zs[l])

        # calculate dbs and dWs
        for l in range(self.L-1, 0, -1):
            self.dbs[l] = self.deltas[l]
            self.dWs[l] = self.deltas[l] @ self.As[l-1].T


    def fit(self,
            X,
            Y,
            epochs=10_000,
            alpha=1e-2,
            printProg=False,
            lambd=1e-2
        ):
        for epoch in range(epochs):
            # get all activations, weighted inputs, and gradients
            self.forward_propagate(X)
            self.back_propagate(Y)

            # gradient descent update
            for l in range(1, self.L):
                self.Ws[l] -= alpha*self.dWs[l]
                self.bs[l] -= alpha*self.bs[l]

            # calculate loss
            loss = self.cross_entropy_loss(X, Y, lambd=lambd)
            if epoch % 1_000 == 0 and printProg:
                print(f"Loss for epoch {epoch} = {loss}")