import math
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def dsigmoid(x):
  return (1 - np.power(x, 2))


def relu(x):
	np.maximum(x, 0, x)

def drelu(x):
	np.where(x > 0, 1, 0)



def relu_backward(dA, activation_cache):
	return dA * drelu(activation_cache)


def sigmoid_backward(dA, activation_cache):
	return dA * dsigmoid(activation_cache)



