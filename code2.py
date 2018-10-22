
import random
from neuron import *
from misc import get_training_data, get_alternate_data
from copy import deepcopy
import matplotlib.pyplot as plt

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
learning_rate = 0.01
momentum = 0.6
squareError = []
epochCount = 0.0
sumSquareError = [0]
SSEStorePermanent = []

def __init__():
    create_layer0_neurons()
    create_layer1_neurons()
    create_layer2_neurons()
 
def create_layer0_neurons():
    neuronListL0.clear()
    #neuronListL0 = []
    n00 = neuron()
    n00.set_weights([1])
    n00.set_bias(0)
 
    n01 = neuron()
    n01.set_weights([1])
    n01.set_bias(0)
 
    n02 = neuron()
    n02.set_weights([1])
    n02.set_bias(0)
    neuronListL0.append(n00)
    neuronListL0.append(n01)
    neuronListL0.append(n02)
 
def create_layer1_neurons():
    neuronListL1.clear()
    n10 = neuron()
    n10.layer = 1
    n10.number = 0
    n10.get_weight_params()
    n10.get_bias()
    n10.add_input_neurons(neuronListL0)
    neuronListL1.append(n10)
 
    n11 = neuron()
    n11.layer = 1
    n11.number =1
    n11.get_weight_params()
    n11.get_bias()
    n11.add_input_neurons(neuronListL0)
    neuronListL1.append(n11)
 
    n12 = neuron()
    n12.layer = 1
    n12.number =2
    n12.get_weight_params()
    n12.get_bias()
    n12.add_input_neurons(neuronListL0)
    neuronListL1.append(n12)
 
    n13 = neuron()
    n13.layer = 1
    n13.number = 3
    n13.get_weight_params()
    n13.get_bias()
    n13.add_input_neurons(neuronListL0)
    neuronListL1.append(n13)
 
    n14 = neuron()
    n14.layer = 1
    n14.number = 4
    n14.get_weight_params()
    n14.get_bias()
    n14.add_input_neurons(neuronListL0)
    neuronListL1.append(n14)
 
    n15 = neuron()
    n15.layer = 1
    n15.number = 5
    n15.get_weight_params()
    n15.get_bias()
    n15.add_input_neurons(neuronListL0)
    neuronListL1.append(n15)
 
    n16 = neuron()
    n16.layer = 1
    n16.number = 6
    n16.get_weight_params()
    n16.get_bias()
    n16.add_input_neurons(neuronListL0)
    neuronListL1.append(n16)
 
    n17 = neuron()
    n17.layer = 1
    n17.number = 7
    n17.get_weight_params()
    n17.get_bias()
    n17.add_input_neurons(neuronListL0)
    neuronListL1.append(n17)
 
    n18 = neuron()
    n18.layer = 1
    n18.number = 8
    n18.get_weight_params()
    n18.get_bias()
    n18.add_input_neurons(neuronListL0)
    neuronListL1.append(n18)
 
    n19 = neuron()
    n19.layer = 1
    n19.number = 9
    n19.get_weight_params()
    n19.get_bias()
    n19.add_input_neurons(neuronListL0)
    neuronListL1.append(n19)
 
    n110 = neuron()
    n110.layer = 1
    n110.number = 10
    n110.get_weight_params()
    n110.get_bias()
    n110.add_input_neurons(neuronListL0)
    neuronListL1.append(n110)
 
def create_layer2_neurons():
    neuronListL2.clear()
    n20 = neuron()
    n20.layer = 2
    n20.number = 0
    n20.get_weight_params()
    n20.get_bias()
    n20.add_input_neurons(neuronListL1)
    neuronListL2.append(n20)
 
    n21 = neuron()
    n21.layer = 2
    n21.number = 1
    n21.get_weight_params()
    n21.get_bias()
    n21.add_input_neurons(neuronListL1)
    neuronListL2.append(n21)
 
def feed_forward(x):
    #Object to store the inputs received
    _input = []
    #Convert the data read to float
    for i in range(len(x)):
        _input.append(float(x[i]))
 
    #Get the output of the layer 1 neurons
    for j in range(len(neuronListL1)):
        layer1Outputs.append(neuron().sigmoid_function(neuronListL1[j].calculate_local_field(_input)))
    #get the output of layer 2 neurons
    for k in range(len(neuronListL2)):
        layer2Outputs.append(neuron().sigmoid_function(neuronListL2[k].calculate_local_field(layer1Outputs)))
 
def compute_error(x):
    _correct_output = []
    #Convert the data read to float
    for i in range(len(x)):
        _correct_output.append(float(x[i]))
 
    for i in range(len(layer2Outputs)):
        errors.append(_correct_output[i] - layer2Outputs[i])
     
def compute_local_gradient(x):
    #Object to store the inputs received
    _input = []
    #Convert the data read to float
    for i in range(len(x)):
        _input.append(float(x[i]))
    #Calculate the local gradient for the output neurons
    for j in range(len(neuronListL2)):
        #The sigmoid Prime value for an is gotten from applying the sigmoid_function_prime to the same
        #input as was fed into the feedforward network
        sigmoidPrimeValue = neuron().sigmoid_function_prime(layer2Outputs[j])
        #Multiply the sigmoid Prime value by the errors computed for a neuron
        localGradientL2.append(errors[j] * sigmoidPrimeValue)
    for k in range(len(neuronListL1)):
        #The sigmoid Prime value for a hidden layer is gotten the same way as it is for a outer layer neuron
        sigmoidPrimeValue = neuron().sigmoid_function_prime(layer1Outputs[k])
        sumOutputLocalGradients = 0
        #Multiply the sigmoid Prime Value by the sum of the local gradients of the output layer
        for _n in range(len(neuronListL2)):
            sumOutputLocalGradients += neuronListL2[_n].weights[k]* localGradientL2[_n]
        localGradientL1.append(sumOutputLocalGradients * sigmoidPrimeValue)
 
def calculate_weight_changes(sample):
    #Write the old weights to a list that backs them up, instantiate the current weight 
    #change as a new list
         
    previousWeightChangeL2 = deepcopy(weightChangeL2)
    previousWeightChangeL1 = deepcopy(weightChangeL1)
    weightChangeL1.clear()
    weightChangeL2.clear()
    ##Set all the previous weight changes to 0
    ##If there are no weight changes in L2, there are no weight changes in L1 
    ##as well so all values in both lists to 0
    if(len(previousWeightChangeL2)==0):
        for _j in range(len(neuronListL2)):
            tempPrevWeightChangeL2 = []
            for _k in range(len(neuronListL2[_j].weights)):
                tempPrevWeightChangeL2.append(0)
            previousWeightChangeL2.append(tempPrevWeightChangeL2)
 
        for _j in range(len(neuronListL1)):
            tempPrevWeightChangeL1 = []
            for _k in range(len(neuronListL1[_j].weights)):
                tempPrevWeightChangeL1.append(0)
            previousWeightChangeL1.append(tempPrevWeightChangeL1)
         
    for i in range(len(neuronListL2)):
        weightChangeL2Temp = []     
        curPrevWeightChangeL2 = previousWeightChangeL2[i]
        #I need to use the y of the ith layer not that of the jth layer.
        for _j in range(len(neuronListL2[i].weights)):
            momentumDelta = momentum * curPrevWeightChangeL2[_j]
            naturalDelta =  learning_rate * localGradientL2[i] * layer1Outputs[_j]
            weightChangeL2Temp.append(momentumDelta + naturalDelta)
        #naturalDelta = learning_rate * localGradientL2[i] * layer2Outputs[i]
        #print("2: " , naturalDelta)
        weightChangeL2.append(weightChangeL2Temp)
 
    for i in range(len(neuronListL1)):
        weightChangeL1Temp = []     
        curPrevWeightChangeL1 = previousWeightChangeL1[i]
        #I need to use the y of the ith layer not that of the jth layer.
        for _j in range(len(neuronListL1[i].weights)):
            momentumDelta = momentum * curPrevWeightChangeL1[_j]
            naturalDelta =  learning_rate * localGradientL1[i] * float(sample[_j])
            weightChangeL1Temp.append(momentumDelta + naturalDelta)
        #naturalDelta = learning_rate * localGradientL2[i] * layer2Outputs[i]
        #print("2: " , naturalDelta)
        weightChangeL1.append(weightChangeL1Temp)
 
def calculate_bias_changes():
    previousBiasChangeL1 = deepcopy(biasChangeL1)
    previousBiasChangeL2 = deepcopy(biasChangeL2)
    biasChangeL2.clear()
    biasChangeL1.clear()
 
    if(len(previousBiasChangeL2)==0):
        for _j in range(len(neuronListL2)):
            previousBiasChangeL2.append(0)
        for _k in range(len(neuronListL1)):
            previousBiasChangeL1.append(0)
 
    for l in range(len(neuronListL2)):
        #print(len(previousBiasChangeL2))
        momentumDelta = momentum * previousBiasChangeL2[l]
        naturalDelta = learning_rate * localGradientL2[l] * 1
        biasChangeL2.append(momentumDelta + naturalDelta)
 
    for m in range(len(neuronListL1)):
        momentumDelta = momentum *previousBiasChangeL1[m]
        naturalDelta = learning_rate *localGradientL1[m] * 1
        biasChangeL1.append(momentumDelta + naturalDelta)
 
def change_weights():
    #print(neuronListL1[9].weights[0])
    for i in range(len(neuronListL2)):
        for j in range(len(neuronListL2[i].weights)):
            neuronListL2[i].weights[j] = round(neuronListL2[i].weights[j] + weightChangeL2[i][j], 4)
     
    for k in range(len(neuronListL1)):
        for l in range(len(neuronListL1[i].weights)):
            neuronListL1[k].weights[l] = round(neuronListL1[k].weights[l] + weightChangeL1[k][l], 4)
 
def change_bias():
    #print("LOL", len(biasChangeL2), "  ",neuronListL2[0].bias, " ", biasChangeL2 )
    for i in range(len(neuronListL2)):
        neuronListL2[i].bias = round(neuronListL2[i].bias + biasChangeL2[i],4)
     
    for j in range(len(neuronListL1)):
        neuronListL1[j].bias = round(neuronListL1[j].bias + biasChangeL1[j], 4)
 
def run_epoch(doShuffle=False, crossValidation =False, i =[]):
    squareError=[]
    if len(i==0):
        i = list(range(314))
        random.shuffle(i)
    if doShuffle:
        print("Shuffle Shuffle")
        for j in i:
            sample = get_training_data(j).split(',')[:3]
            output = get_training_data(j).split(',')[3:]
            feed_forward(sample)
            #print("sample ", j, " running")
            #print("layer 1 Output is: ",layer1Outputs)
            #print("layer 2 Output is: ", layer2Outputs)
            compute_error(output)
            compute_local_gradient(sample)
            calculate_weight_changes(sample)
            change_weights()
            calculate_bias_changes()
            change_bias()
            #print("The errors are: ", errors)
            squareError.append(sum(errors)**2)
            #print(squareError[-1])
            layer1Outputs.clear()
            layer2Outputs.clear()
            localGradientL1.clear()
            localGradientL2.clear()
            errors.clear()
    if not(doShuffle) and not(crossValidation):
        for j in range(314):
            sample = get_training_data(j).split(',')[:3]
            output = get_training_data(j).split(',')[3:]
            feed_forward(sample)
            #print("sample ", j, " running")
            #print("layer 1 Output is: ",layer1Outputs)
            #print("layer 2 Output is: ", layer2Outputs)
            compute_error(output)
            compute_local_gradient(sample)
            calculate_weight_changes(sample)
            change_weights()
            calculate_bias_changes()
            change_bias()
            #print("The errors are: ", errors)
            squareError.append(sum(errors)**2)
            #print(squareError[-1])
            layer1Outputs.clear()
            layer2Outputs.clear()
            localGradientL1.clear()
            localGradientL2.clear()
            errors.clear()
    
    sumSquareError.append(sum(squareError))

def reinitialize_network(lrate, momentum):
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
    learning_rate = lrate
    momentum = momentum
    squareError = []
    epochCount = 0.0
    sumSquareError = [0]

def train_network():
    epochCount = 1
    run_epoch()
    #print(epochCount, "with difference ", sumSquareError[-2]-sumSquareError[-1])
    while abs(sumSquareError[-2]-sumSquareError[-1]) > 0.001:
        run_epoch(True, False, [])
        print(epochCount, "with difference ", sumSquareError[-2]-sumSquareError[-1])
        epochCount +=1
    SSEStorePermanent.append(sumSquareError)
    return epochCount


__init__()

print(train_network())

LR001 = [0.3428190600000003, 0.29114563000000043, 0.2513126000000001, 0.22314984, 0.20491820000000005, 0.18441787999999998, 0.16088408000000012, 0.1541833300000001, 0.14379660999999985, 0.12977433, 0.12183912999999998, 0.11041355000000011, 0.10230268999999995, 0.09500266000000002, 0.09113095000000003, 0.08194340000000001, 0.07975074999999995, 0.07220497000000008, 0.06985128999999998, 0.06596641999999998, 0.06224890000000004, 0.060318560000000014, 0.057125390000000005, 0.05597069000000001, 0.05225354000000005, 0.05065231, 0.048708320000000006, 0.047417230000000005, 0.04691909]

LR07 = [0.17412999000000015, 0.2063402100000001, 0.025554800000000013, 0.005099339999999996, 0.001848639999999997, 0.0009220600000000008]

LR02 = [0.1728649700000001, 0.05432344000000004, 0.05646818999999997, 0.08948851999999993, 0.15885820000000014, 0.12866524000000001, 0.06887550000000005, 0.03558869, 0.02031809999999997, 0.012220259999999995, 0.008176079999999992, 0.005398379999999998, 0.0038096600000000025, 0.0028294800000000027]

LR09 = [0.21833668000000006, 0.10937024999999999, 0.00871059000000001, 0.001911540000000004, 0.0009401800000000004]

#Plot the difference in convergence using the different values of the 
plt.plot(LR001)
plt.plot(LR02)
plt.plot(LR07)
plt.plot(LR09)
plt.xlabel("NO. Of Epochs")
plt.ylabel("SSE")
plt.legend(('learning_rate = 0.01', 'learning_rate = 0.2', 'learning_rate=0.7', 'learning_rate=0.9'), loc = "upper right")
plt.show()

M0 = [0, 0.3487129699999999, 0.31725829999999994, 0.2784896499999999, 0.25258549999999996, 0.2362698100000001, 0.2146182500000001, 0.20104141999999972, 0.18844249, 0.1777006, 0.16776181999999998, 0.15817248000000012, 0.15306556999999993, 0.14070031999999996, 0.13569449, 0.13167308000000005, 0.12244747999999994, 0.11626573000000004, 0.11312757000000005, 0.1063972, 0.10034563, 0.09523428999999996, 0.09190269, 0.08913997, 0.08413930999999997, 0.08067421000000013, 0.0774314700000001, 0.07457432999999997, 0.07394254999999995]

M06 = [0, 0.3264504499999999, 0.2520159399999998, 0.20007317000000005, 0.16826175999999995, 0.14664049000000004, 0.12944406999999988, 0.11297603000000002, 0.10041909000000003, 0.08839543999999988, 0.07772261999999998, 0.06964125, 0.06449923000000003, 0.058346079999999995, 0.053983270000000014, 0.051525620000000015, 0.04840357999999997, 0.04751635000000004]

plt.plot(M0)
plt.plot(M06)
plt.xlabel("NO. Of Epochs")
plt.ylabel("SSE")
plt.legend(('Momentum = 0', 'Momentum = 0.6'), loc = "upper right")
plt.show()