from math import e 
class neuron(object):
    def __init__(self):
        self.weights = []
        self.number = 0
        self.layer = 0
        self.bias = 0
        self.inputNeurons = []
    
    def set_weights(self, weightsList):
        for i in weightsList:
            self.weights.append(float(i))
    
    def set_bias(self, bias):
        self.bias = float(bias)

    def calculate_local_field(self, x):
        local_field = 0
        #print(len(self.inputNeurons))
        if(len(self.inputNeurons) == 0):
            print("We're working with an input neuron")
            local_field = x[0]
        else:
            for i in range(len(x)):
                #print(i, " ", self.weights, " ", len(self.weights), " ", len(x), x)
                #print(self.weights[i])
                local_field += x[i] * self.weights[i]
        return local_field + self.bias

    def sigmoid_function(self, x):
        _retVal = 1 / (1 + e**-x)
        return(_retVal)

    def sigmoid_function_prime(self, x):
        return (e**-x)/(1 + e**-x)**2

    def add_input_neurons(self, inputNeuronList):
        for i in range(len(inputNeuronList)):
            self.inputNeurons.append(inputNeuronList[i])

    def get_weight_params(self):
        if self.layer == 1:
            file = open(".\params\w1.csv", 'r')
        if self.layer == 2:
            file = open(".\params\w2.csv", 'r')
        self.set_weights(file.readlines()[self.number].rstrip().split(','))

    def get_bias(self):
        if self.layer == 1:
            file = open(".\params\B1.csv", 'r')
        if self.layer == 2:
            file = open(".\params\B2.csv", 'r')
        self.set_bias(file.readlines()[self.number].rstrip())
       #print(self.layer, self.number)