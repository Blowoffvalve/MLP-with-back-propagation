def get_training_data(k):
    file = open("./params/cross_data.csv", "r")
    return file.readlines()[k].rstrip()

def get_alternate_data(k):
    file = open("./params/Two_Class_FourDGaussians500.txt", "r")
    return file.readlines()[k].rstrip()