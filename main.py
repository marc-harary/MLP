from NN import NN
import numpy as np

def main():
    nn = NN(layers=[5,3,2,1],
            activations=["ReLU", "ReLU", "ReLU", "sigmoid"])
    X = np.random.random((5,2))
    nn.forward_propagate(X)
    print(nn.As[-1])

if __name__ == "__main__":
    main()