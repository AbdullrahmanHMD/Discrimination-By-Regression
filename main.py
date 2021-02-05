import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math
import pandas as pd

np.random.seed(38)

def safelog(x):
    return(np.log(x + 1e-100))
#-------------------------------------------------------------------------------
#--------------------------------Initializing Data------------------------------
def get_data(data_lables):
    data = []
    for i in range(len(data_lables)):
        data.append(data_lables[i][1])
    return data
    
data_labels = np.genfromtxt("hw02_data_set_labels.csv", dtype = str, delimiter = ',')
data_images = np.genfromtxt("hw02_data_set_images.csv", dtype = int, delimiter = ',')

num_of_classes = 5

data_labels = get_data(data_labels)
data_set = np.array([(ord(x) - 65) for x in data_labels])
#onehot_lables = np.eye(5)[data_set]
#-------------------------------------------------------------------------------
#--------------------------------Dividing Data----------------------------------
def get_sets(array):
    train_set, test_set , training_labels, test_labels = [], [], [], []
    count = 0
    for i in range(len(array)):
        if count >= 39:
            count = 0
        if count < 25:
            train_set.append(array[i])
            training_labels.append(data_set[i])
            count += 1
        elif count >= 25 and count < 39:
            test_set.append(array[i])
            test_labels.append(data_set[i])
            count += 1
    return np.array(test_set) ,np.array(train_set),np.array(training_labels), np.array(test_labels)

test_set, training_set, training_labels, test_labels = np.array(get_sets(data_images))


onehot_encoded_lables = np.zeros(shape = (125,5))
onehot_encoded_lables[range(125), training_labels] = 1
#-------------------------------------------------------------------------------
#--------------------------------Gradiant Decent Functions----------------------
def sigmoid_function(x, w, w_0):
    term = np.matmul(w, x) + w_0
    return  1/(1 + np.exp(-term))

def gradiant_decent_w(x, y_truth, y_predicted):
    arr = []
    for i in range(num_of_classes):
        mat = np.zeros(x.shape[1], dtype = float)
        for j in range(len(y_truth)):
            mat += -((y_truth[j][i] - y_predicted[j][i]) * y_predicted[j][i] * (1 - y_predicted[j][i])) * x[j] # 1 x 320
        arr.append(mat)
    return np.array(arr)

def gradiant_decent_w0(y_truth, y_predicted):
    arr = []
    for i in range(num_of_classes):
        mat = 0.0
        for j in range(len(y_truth)):
            mat += -((y_truth[j][i] - y_predicted[j][i]) * y_predicted[j][i] * (1 - y_predicted[j][i]))
        arr.append(mat)
    return np.array(arr)
#-------------------------------------------------------------------------------
#--------------------------------Parameters Estimation--------------------------
eta = 0.01
epsilon = 1e-3

w = np.random.uniform(-0.01, 0.01, (num_of_classes,data_images.shape[1]))
w_0 = np.random.uniform(-0.01, 0.01, num_of_classes)

def predicted_labels(data, W, W_0): # W.shape = (5, 320) W_0.shape = (5,)
    mat = []
    k = len(data)
    for i in range(k):
        arr = [sigmoid_function(data[i], W[j], W_0[j]) for j in range(num_of_classes)]
        mat.append(arr)
    return np.array(mat)

def error_quantity(w, w_prev, w_0, w0_prev):
    term1, term2 = np.sum((w_0 - w0_prev)**2), np.sum((w - w_prev)**2)
    return np.sqrt(term1 + term2)

def c_parameters(lables, images, w, w_0):
    iter = 0
    objective_values = []
    while True:
        y_pred = predicted_labels(images, w, w_0)
        
        objective_values = np.append(objective_values, -np.sum(lables * safelog(y_pred)))

        w_prev = w
        w0_prev = w_0
        
        w = w - eta * gradiant_decent_w(images, lables, y_pred)
        w_0 = w_0 - eta * gradiant_decent_w0(lables, y_pred)
        
        error = error_quantity(w, w_prev, w_0, w0_prev)
        
        if error < epsilon:
            break
        iter += 1
    return objective_values, w, w_0
#-------------------------------------------------------------------------------
#--------------------------------Confusion Matrices and Data Plotting-----------
objective_values, w, w_0 = c_parameters(onehot_encoded_lables, training_set, w, w_0)
y_predicted = predicted_labels(training_set, w , w_0)
max = np.argmax(y_predicted, axis = 1)
confusion_matrix = pd.crosstab(max, training_labels, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)

y_predicted = predicted_labels(test_set, w , w_0)
max = np.argmax(y_predicted, axis = 1)
confusion_matrix = pd.crosstab(max, test_labels, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)

plt.plot(objective_values)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()