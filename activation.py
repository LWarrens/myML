import numpy as np
class ActivationFunction(object):
  activation_type = None
  def __init__(self, function, function_derivative):
    self.function = function
    self.function_derivative = function_derivative

class LinearActivation(ActivationFunction):
  def __init__(self):
    ActivationFunction.__init__(self, lambda x : x, lambda x: 1)

class TanhActivation(ActivationFunction):
  def __init__(self):
    ActivationFunction.__init__(self, np.tanh, (lambda x: 1. - np.square(np.tanh(x))))

class LogisticActivation(ActivationFunction):
  def __init__(self):
    ActivationFunction.__init__(self, (lambda x: 1. / (1. + np.exp(-x))), (lambda x: np.exp(-x) / np.square(1. + np.exp(-x))))

class SoftplusActivation(ActivationFunction):
  def __init__(self):
    ActivationFunction.__init__(self, (lambda x: np.log(1. + np.exp(x))), (lambda x: 1. / (1. + np.exp(x))))

class RectifierActivation(ActivationFunction):
  def __init__(self):
    ActivationFunction.__init__(self, (lambda x: np.max(0, x)), (lambda x: np.max(0, np.sign(x))))

class LeakyRectifierActivation(ActivationFunction):
  activation_type = "parametric"
  def __init__(self):
    ActivationFunction.__init__(self, (lambda x: np.max(0, x)), (lambda x: np.max(0, np.sign(x))))