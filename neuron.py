class neuron(object):
    def __init__(self):
        self.weights = []
        self.number = 0
        self.layer = 0
        self.bias = 0
        self.inputNeurons = []
    
    def set_weights(self, weightsList):
        for i in weightsList:
            self.weights.append(i)
    
    def set_bias(self, bias):
        self.bias = bias

    def calculate_local_field(self, x):
        local_field = 0
        print(len(self.inputNeurons))
        if(len(self.inputNeurons) == 0):
            print("We're working with an input neuron")
            local_field = x[0]
        else:
            for i in range(len(x)):
                print(x[i] * self.weights[i])
                local_field += x[i] * self.weights[i]
        return local_field + self.bias
        
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
n10.inputNeurons.append(n00)
n10.inputNeurons.append(n01)
n10.inputNeurons.append(n02)
print(n10.calculate_local_field([-1.992, -1.679, -0.068]))
