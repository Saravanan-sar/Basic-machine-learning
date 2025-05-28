
from numpy import exp, array, random, dot

class NeuralNetwork():
    def _init_(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs, iterations):
        for _ in range(iterations):
            output = self.think(inputs)
            error = outputs - output
            adjustments = dot(inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        return self.sigmoid(dot(inputs, self.synaptic_weights))

nn = NeuralNetwork()
inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputs = array([[0, 1, 1, 0]]).T
nn.train(inputs, outputs, 10000)
print(nn.think(array([1, 0, 0])))
