"""
implementation of backpropagation network as described in
http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
"""
import sklearn
import numpy as np
from activation import TanhActivation

def makenumpyarray(x):
  return np.array(x) if type(x) is list else np.array([x]) if type(x) is not np.ndarray else x

class Neuron():
  def __init__(self, ninputs=0, activation = TanhActivation):
    self.activation = activation()
    self.bias = np.random.rand() - .5
    self.weights = np.random.rand(ninputs) - .5

  def activate(self, neuron_input):
    activation = self.bias + np.dot(self.weights, neuron_input)
    output = self.activation.function(activation)
    output_derivative = self.activation.function_derivative(activation)
    return output, output_derivative

  def update(self, learning_rate, delta, neuron_input):
    change = learning_rate * delta
    self.weights += change * neuron_input
    self.bias += change

# fully connected layers...for now
class Layer():
  def __init__(self, size = 0, activation = TanhActivation):
    self.neurons = []
    for i in range(size):
      self.neurons.append(Neuron(activation=activation))
  def add(self, size, activation = TanhActivation):
    for i in range(size):
      self.neurons.append(Neuron(activation=activation))
    return self

class PerceptronNetwork():
  def __init__(self, ninputs = 1, noutputs = 1, layers = None):
    self.ninputs = ninputs
    self.noutputs = noutputs
    self.layers = [] if not layers else layers

  def finalize(self):
    if self.noutputs != len(self.layers[-1].neurons):
      raise Exception("The last layer in the network must be the size of outputs")
    for neuron in self.layers[0].neurons:
      neuron.weights = np.random.rand(self.ninputs) - .5
    for layer_index in range(1, len(self.layers)):
      for neuron in self.layers[layer_index].neurons:
        neuron.weights = np.random.rand(len(self.layers[layer_index - 1].neurons)) - .5

  def appendLayer(self, layer):
    self.layers.append(layer)
  def popLayer(self):
    self.layers.pop(len(self.layers))

  def propagate(self, input_data, dropout = 0):
    input_data = makenumpyarray(input_data)
    if input_data.ndim > 2:
      raise Exception("Example data should not have more than 2 dimensions")
    if input_data.ndim == 2 and input_data.shape[1] != self.ninputs:
      raise Exception("Input size is not consistent with the network")
    if input_data.ndim == 1 and 1 != self.ninputs:
      raise Exception("Input size is not consistent with the network")
    if input_data.ndim is 0:
      raise Exception("Must input valid sample(s)")

    depth = len(self.layers)
    output = np.array(map(lambda sample: map(lambda layer: np.zeros(len(layer.neurons)), self.layers), input_data))
    output_derivatives = np.array(map(lambda sample: map(lambda layer: np.zeros(len(layer.neurons)), self.layers), input_data))

    for sample_index in range(input_data.shape[0]):
      output[sample_index], output_derivatives[sample_index] = self.__propagate(input_data[sample_index], dropout)
    return output[:, depth - 1], output, output_derivatives

  # single instance propagation method
  def __propagate(self, sample, dropout = 0):
    input_layer = 0
    output = np.array(map(lambda layer: np.zeros(len(layer.neurons)), self.layers))
    output_derivatives = np.array(map(lambda layer: np.zeros(len(layer.neurons)), self.layers))
    for neuron_index, neuron in enumerate(self.layers[input_layer].neurons):
      if np.random.rand() > dropout or input_layer is (len(self.layers) - 1):
        output[input_layer][neuron_index], output_derivatives[input_layer][neuron_index] = neuron.activate(sample)
    for layer_index in range(1, len(self.layers)):
      if np.random.rand() > dropout or layer_index is (len(self.layers) - 1):
        last_layer_output = output[layer_index - 1]
        for neuron_index, neuron in enumerate(self.layers[layer_index].neurons):
          output[layer_index][neuron_index], output_derivatives[layer_index][neuron_index] = neuron.activate(last_layer_output)
    return output, output_derivatives

  def backpropagate(self, samples, target, learning_rate = .1, weight_decay = .5, dropout = 0):
    samples, target = makenumpyarray(samples), makenumpyarray(target)
    if samples.ndim != target.ndim:
      raise Exception("The sample(s) and label(s) should have the same number of dimensions")
    if samples.ndim > 2:
      raise Exception("Each sample must be a 1-D array")

    depth = len(self.layers)


    """
      Step #1: Propagate
    """
    actual_output, outputs, output_derivatives = self.propagate(samples, dropout)
    """
      Step #2: Backpropagate
    """
    output_layer = depth - 1
    learning_rate /= samples.shape[0]
    error = np.array(map(lambda sample: map(lambda layer: np.zeros(len(layer.neurons)), self.layers), samples))
    for sample_index in range(samples.shape[0]):
      error[sample_index] = self.__backpropagate(actual_output[sample_index], target[sample_index])
    """
      Step #3: Update all neurons in network
      update input layer with the sample as input
    """
    for sample_index in range(samples.shape[0]):
      for neuron_index, neuron in enumerate(self.layers[0].neurons):
        delta = error[sample_index, 0][neuron_index] * output_derivatives[sample_index, 0][neuron_index]
        neuron.update(learning_rate, delta, samples[sample_index])
      for layer_index in range(1, depth):
        for neuron_index, neuron in enumerate(self.layers[layer_index].neurons):
          delta = error[sample_index, layer_index][neuron_index] * output_derivatives[sample_index, layer_index][neuron_index]
          neuron.update(learning_rate, delta, outputs[sample_index, layer_index - 1])

    # total_error = np.mean((target - actual_output)**2, 0)
    # print "total error:", np.sum(total_error)
    
  # single instance backpropagation method
  def __backpropagate(self, actual_output, target_output):
    sample_error = np.array(map(lambda layer: np.zeros(len(layer.neurons)), self.layers))
    # simple backpropagation over each output node
    difference = target_output - actual_output
    # update the error of the final nodes
    output_layer = len(self.layers) - 1
    sample_error[output_layer] = difference
    # go backwards over every other node
    for layer_index in reversed(range(output_layer)):
      for neuron_index, neuron in enumerate(self.layers[layer_index].neurons):
        next_layer = layer_index + 1
        # get weights that correspond to this neuron's output
        next_layer_weights = np.array(map(lambda nl_neuron: nl_neuron.weights[neuron_index], \
                                          self.layers[next_layer].neurons))
        sample_error[layer_index][neuron_index] = np.sum(next_layer_weights * sample_error[next_layer])
    return sample_error

  def predict(self, sample_set):
    actual, outputs, outputs_derivative = self.propagate(sample_set)
    return actual

  def get_weights(self):
    return np.array(map(lambda layer: map(lambda neuron: neuron.weights, layer.neurons), self.layers))

  def get_biases(self):
    return np.array(map(lambda layer: map(lambda neuron: neuron.bias, layer.neurons), self.layers))
