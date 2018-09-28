from neuron import *
from misc import get_training_data
from copy import deepcopy

class network:
    neuronListL0 = []
    neuronListL1 = []
    neuronListL2 = []
    layer1Outputs = []
    layer2Outputs = []
    localGradientL1 =[]
    localGradientL2 =[]
    weightChangeL1 = []
    previousWeightChangeL1 = []
    previousBiasChangeL1 = []
    biasChangeL1=[]
    weightChangeL2 = []
    previousWeightChangeL2 = []
    previousBiasChangeL2 = []
    biasChangeL2 = []
    errors = []
    learning_rate = 0.7
    momentum = 0.3

    def __init__(self):
        self.create_layer0_neurons()
        self.create_layer1_neurons()
        self.create_layer2_neurons()

    def create_layer0_neurons(self):
        self.neuronListL0.clear()
        #self.neuronListL0 = []
        n00 = neuron()
        n00.set_weights([1])
        n00.set_bias(0)

        n01 = neuron()
        n01.set_weights([1])
        n01.set_bias(0)

        n02 = neuron()
        n02.set_weights([1])
        n02.set_bias(0)
        self.neuronListL0.append(n00)
        self.neuronListL0.append(n01)
        self.neuronListL0.append(n02)

    def create_layer1_neurons(self):
        self.neuronListL1.clear()
        n10 = neuron()
        n10.layer = 1
        n10.number = 0
        n10.get_weight_params()
        n10.get_bias()
        n10.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n10)

        n11 = neuron()
        n11.layer = 1
        n11.number =1
        n11.get_weight_params()
        n11.get_bias()
        n11.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n11)

        n12 = neuron()
        n12.layer = 1
        n12.number =2
        n12.get_weight_params()
        n12.get_bias()
        n12.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n12)

        n13 = neuron()
        n13.layer = 1
        n13.number = 3
        n13.get_weight_params()
        n13.get_bias()
        n13.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n13)

        n14 = neuron()
        n14.layer = 1
        n14.number = 4
        n14.get_weight_params()
        n14.get_bias()
        n14.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n14)

        n15 = neuron()
        n15.layer = 1
        n15.number = 5
        n15.get_weight_params()
        n15.get_bias()
        n15.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n15)

        n16 = neuron()
        n16.layer = 1
        n16.number = 6
        n16.get_weight_params()
        n16.get_bias()
        n16.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n16)

        n17 = neuron()
        n17.layer = 1
        n17.number = 7
        n17.get_weight_params()
        n17.get_bias()
        n17.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n17)

        n18 = neuron()
        n18.layer = 1
        n18.number = 8
        n18.get_weight_params()
        n18.get_bias()
        n18.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n18)

        n19 = neuron()
        n19.layer = 1
        n19.number = 9
        n19.get_weight_params()
        n19.get_bias()
        n19.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n19)

        n110 = neuron()
        n110.layer = 1
        n110.number = 10
        n110.get_weight_params()
        n110.get_bias()
        n110.add_input_neurons(self.neuronListL0)
        self.neuronListL1.append(n110)

    def create_layer2_neurons(self):
        self.neuronListL2.clear()
        n20 = neuron()
        n20.layer = 2
        n20.number = 0
        n20.get_weight_params()
        n20.get_bias()
        n20.add_input_neurons(self.neuronListL1)
        self.neuronListL2.append(n20)

        n21 = neuron()
        n21.layer = 2
        n21.number = 1
        n21.get_weight_params()
        n21.get_bias()
        n21.add_input_neurons(self.neuronListL1)
        self.neuronListL2.append(n21)

    def feed_forward(self, x):
        #Object to store the inputs received
        _input = []
        #Convert the data read to float
        for i in range(len(x)):
            _input.append(float(x[i]))

        #Get the output of the layer 1 neurons
        for j in range(len(self.neuronListL1)):
            self.layer1Outputs.append(neuron().sigmoid_function(self.neuronListL1[j].calculate_local_field(_input)))
        #get the output of layer 2 neurons
        for k in range(len(self.neuronListL2)):
            self.layer2Outputs.append(neuron().sigmoid_function(self.neuronListL2[k].calculate_local_field(self.layer1Outputs)))

    def compute_error(self, x):
        _correct_output = []
        #Convert the data read to float
        for i in range(len(x)):
            _correct_output.append(float(x[i]))

        for i in range(len(self.layer2Outputs)):
            self.errors.append(_correct_output[i] - self.layer2Outputs[i])

    def compute_local_gradient(self, x):
        #Object to store the inputs received
        _input = []
        #Convert the data read to float
        for i in range(len(x)):
            _input.append(float(x[i]))
        #Calculate the local gradient for the output neurons
        for j in range(len(self.neuronListL2)):
            #The sigmoid Prime value for an is gotten from applying the sigmoid_function_prime to the same
            #input as was fed into the feedforward network
            sigmoidPrimeValue = neuron().sigmoid_function_prime(self.neuronListL2[j].calculate_local_field(_input))
            #Multiply the sigmoid Prime value by the errors computed for a neuron
            self.localGradientL2.append(self.errors[j] * sigmoidPrimeValue)
        for k in range(len(self.neuronListL1)):
            #The sigmoid Prime value for a hidden layer is gotten the same way as it is for a outer layer neuron
            sigmoidPrimeValue = neuron().sigmoid_function_prime(self.neuronListL1[k].calculate_local_field(_input))
            #Multiply the sigmoid Prime Value by the sum of the local gradients of the output layer
            self.localGradientL1.append(sum(self.localGradientL2) * sigmoidPrimeValue)

    def calculate_weight_changes(self):
        #Write the old weights to a list that backs them up, instantiate the current weight 
        #change as a new list
        self.previousWeightChangeL2 = deepcopy(self.weightChangeL2)
        self.previousWeightChangeL1 = deepcopy(self.weightChangeL1)
        self.weightChangeL1.clear()
        self.weightChangeL2.clear()
        ##Set all the previous weight changes to 0
        ##If there are no weight changes in L2, there are no weight changes in L1 
        ##as well so all values in both lists to 0
        if(len(self.previousWeightChangeL2)==0):
            for j in range(len(self.neuronListL2)):
                self.previousWeightChangeL2.append(0)
            for k in range(len(self.neuronListL1)):
                self.previousWeightChangeL1.append(0)
                
        for i in range(len(self.neuronListL2)):
            momentumDelta = round(self.momentum * self.previousWeightChangeL2[i], 4)
            naturalDelta = round(self.learning_rate * self.localGradientL2[i] * self.layer2Outputs[i], 4)
            self.weightChangeL2.append(momentumDelta + naturalDelta)

        for j in range(len(self.neuronListL1)):
            momentumDelta = round(self.momentum * self.previousWeightChangeL1[j], 4)
            naturalDelta = round(self.learning_rate * self.localGradientL1[j] * self.layer1Outputs[i], 4)
            self.weightChangeL1.append(momentumDelta + naturalDelta)
    
    def calculate_bias_changes(self):
        self.previousBiasChangeL1 = deepcopy(self.biasChangeL1)
        self.previousBiasChangeL2 = deepcopy(self.biasChangeL2)
        self.biasChangeL2.clear()
        self.biasChangeL1.clear()

        if(len(self.previousBiasChangeL2)==0):
            for j in range(len(self.neuronListL2)):
                self.previousBiasChangeL2.append(0)
            for k in range(len(self.neuronListL1)):
                self.previousBiasChangeL1.append(0)

        for l in range(len(self.neuronListL2)):
            #print(len(self.previousBiasChangeL2))
            momentumDelta = round(self.momentum * self.previousBiasChangeL2[l], 4)
            naturalDelta = round(self.learning_rate * self.localGradientL2[l] * 1, 4)
            self.biasChangeL2.append(momentumDelta + naturalDelta)

        for m in range(len(self.neuronListL1)):
            momentumDelta = round(self.momentum *self.previousBiasChangeL1[m], 4)
            naturalDelta = round(self.learning_rate *self.localGradientL1[m] * 1,4)
            self.biasChangeL1.append(momentumDelta + self.localGradientL1[m])

    def change_weights(self):
        #print(self.neuronListL1[9].weights[0])
        for i in range(len(self.neuronListL2)):
            for j in range(len(self.neuronListL2[i].weights)):
                self.neuronListL2[i].weights[j] -= round(self.weightChangeL2[i], 4)
        
        for k in range(len(self.neuronListL1)):
            for l in range(len(self.neuronListL1[i].weights)):
                self.neuronListL1[k].weights[l] -= round(self.weightChangeL1[k], 4)
    
    def change_bias(self):
        #print("LOL", len(self.biasChangeL2), "  ",self.neuronListL2[0].bias, " ", self.biasChangeL2 )
        for i in range(len(self.neuronListL2)):
            self.neuronListL2[i].bias -= round(self.biasChangeL2[i],4)
        
        for j in range(len(self.neuronListL1)):
            self.neuronListL1[j].bias -= round(self.biasChangeL1[j], 4)

    def run_epoch(self, noOfEpochs):
        for i in range(noOfEpochs):
            for j in range(313):
                sample = get_training_data(j).split(',')[:3]
                output = get_training_data(j).split(',')[3:]
                n1.feed_forward(sample)
                
                n1.compute_error(output)
                n1.compute_local_gradient(sample)
                n1.calculate_weight_changes()
                n1.change_weights()
                n1.calculate_bias_changes()
                n1.change_bias()
                self.layer1Outputs.clear()
                self.layer2Outputs.clear()
                self.localGradientL1.clear()
                self.localGradientL2.clear()

n1 = network()
n1.run_epoch(1)
for i, neuron in enumerate(n1.neuronListL1):
    print("Layer 1 Neuron ", i+1, "has weights ", neuron.weights )

for j, neuron in enumerate(n1.neuronListL2):
    print("Layer 2 Neuron ", j+1, "has weights", neuron.weights)