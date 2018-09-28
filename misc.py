def get_training_data(k):
    file = open("./params/cross_data.csv", "r")
    return file.readlines()[k].rstrip()
