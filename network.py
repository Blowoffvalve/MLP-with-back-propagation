from neuron import *

n00 = neuron()
n00.set_weights([1])
n00.set_bias(0)

n01 = neuron()
n01.set_weights([1])
n01.set_bias(0)

n02 = neuron()
n02.set_weights([1])
n02.set_bias(0)

n10 = neuron()
n10.set_weights([0.4033, -1.0562, 0.2306])
n10.set_bias(-0.122)
n10.add_input_neurons([n00, n01, n02])
print(n10.calculate_local_field([-1.992, -1.679, -0.068]))
