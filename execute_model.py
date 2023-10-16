import sys
sys.path.append('C:/Users/Kamil/Desktop/spamFilterAi/data_preparation')

#Taking data and classes to giv to model
from data_preparation.data_loading import data_classes
#Taking vextorized emails [[0,1,0,1],[...],...]
from feature_extraction.Vocabulary import vectorized_emails,vectorized_test_emails
#Model
from perceptron_algorithm.Neuron import  Neuron
from perceptron_algorithm.Neural_network import Perceptron


def accurate_result(classes):
    count_ham = 0
    count_spam = 0
    for clas in classes:
        if clas == 'ham':
            count_ham+=1
        else:
            count_spam+=1
    return count_spam,count_ham

def model_training(dataset_of_vectorized_emails):
    n = len(dataset_of_vectorized_emails)
    emails = dataset_of_vectorized_emails
    list_of_neurons = []
    for i in range(0,n):
        new_neuron = Neuron(input_size=len(emails[i]),id=i)
        list_of_neurons.append(new_neuron)

    num_of_spam, num_of_ham = accurate_result(data_classes)
    neural_network = Perceptron(list_of_neurons=list_of_neurons,learning_rate=0.2, convergence_criteria=0.1)
    neural_network.train(input_data=vectorized_emails,labels=data_classes,list_of_neurons=list_of_neurons)

    prediction = neural_network.prediction(feature_vector=emails, list_of_neurons=list_of_neurons)

    return predictions

trained_model_predictions = model_training(vectorized_emails)
print(trained_model_predictions)