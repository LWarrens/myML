import numpy as np
import sys
import pyqtgraph as pg
from activation import TanhActivation
from activation import LinearActivation
from activation import SoftplusActivation
from perceptron import PerceptronNetwork
from perceptron import Layer
np.random.seed(3)
def BooltoNum(x):
  if type(x) is np.array:
    return x.astype(float)
  return float(x)

# XOR just to show its deep
enet = PerceptronNetwork(ninputs=1, noutputs=1)
#enet.appendLayer(Layer(4, activation=TanhActivation))
# enet.appendLayer(Layer(4, activation=TanhActivation))
enet.appendLayer(Layer(4, activation=TanhActivation))
enet.appendLayer(Layer(4, activation=TanhActivation))
enet.appendLayer(Layer(1, activation=LinearActivation))
enet.finalize()


func = lambda x: np.sin(x/3)

# print "weights", enet.get_weights()
# print "biases", enet.get_biases()

window = pg.GraphicsWindow(title="MLP Example")
wplot = window.addPlot(title="Function")
samples = np.linspace(-10, 10, 100)
ans = map(func, samples)
c1 = wplot.plot(np.array(np.transpose([samples, ans])), pen="c")

epoch = 1000000
print "training in house network",
for j in range(0, epoch):
  if epoch >= 9 and j % (epoch / 9) == 0:
    print ".",
  training_samples = 10 * 2 * (np.random.random_sample() - .5)
  training_answers = func(training_samples)
  enet.backpropagate(training_samples, target=training_answers, learning_rate=.001, dropout = .025)
print " done"

x1 = enet.predict(samples)
# print "weights", enet.get_weights()
# print "biases", enet.get_biases()
print "predict", x1
x_val = np.array(map(lambda x: x[0], x1))
c2 = wplot.plot(np.array(np.transpose([samples, x_val])), pen='r')
wplot.show()

if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
  pg.QtGui.QApplication.exec_()

