from NN import NN
import numpy as np

def main():
    nn = NN(ns=[5,5,1], acts=["ReLU", "ReLU", "sigmoid"])
    X = np.random.random((5,1))
    # print(nn.L)
    output = nn.forward_propagate(X)
    print(output)
    # nn.back_propagate(1)

if __name__ == "__main__":
    main()