import numpy as np


def modSigmoid(x):
    return 2.0 / (1 + np.exp(-2*x)) -1

def modSign(x):
    return (np.sign(x) + 1) / 2.0

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def noFunc(x):
    return x