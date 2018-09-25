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

    def add_input_neurons(self, inputNeuronList):
        for i in range(len(inputNeuronList)):
            self.inputNeurons.append(inputNeuronList[i])