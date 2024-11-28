import numpy as np

# Neuron
class Network:
    def __init__(self, N, alpha):
        self.W1 = np.ones(N+1)
        self.W2 = np.ones(N+1)
        self.alpha = alpha

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def predict(self, X):
        X = np.atleast_2d(X)
        X = np.c_[X, np.ones(1)]
        p = self.sigmoid(X @ self.W)
        print(p)
    
    def training(self, X, y, iters):
        X = np.c_[X, np.ones(X.shape[0])]
        for i in range(iters):
            for (x, Y_) in zip(X, y):
                Y = self.sigmoid(x @ self.W)
                if Y != Y_:
                    error = Y - Y_
                    self.W += -self.alpha*error*x
    
net = Network
x = np.array([1,2])
net.predict()