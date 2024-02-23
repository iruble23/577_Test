from single_neuron.activations import sigmoid_activation
from single_neuron.loss_functions import cross_entropy_loss

print(cross_entropy_loss(sigmoid_activation(0.5), 1))