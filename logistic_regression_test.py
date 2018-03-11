import CustomAI as jarvis_jr
import numpy as np
from Enumerations import Regressions

# Step 1    load data set
foo = open("Diagnostic_breast_cancer.txt", "r")

file = foo.read().split('\n')
foo.close()

# Step 2    fix the data format as : matrix[m][n]
# inputs 1: param[1] param[2] param[3] ... param[n]
# inputs 2: param[1] param[2] param[3] ... param[n]
# ...
# inputs m: param[1] param[2] param[3] ... param[n]

arr = []
test = np.array(file)

for t in range(test.shape[0]):
    arr.append(test[t].split(','))

test = np.array(arr)

# Step 3    change output value letter --> number {0,1}
for t in range(test.shape[0]):
    for k in range(test.shape[1]):
        if test[t][k] == 'B':
            test[t][k] = '0'
        elif test[t][k] == 'M':
            test[t][k] = '1'

# Step 4    our data set contains id column so we must extract it first
test = test[:, 1:]      # extract id

data_set = test[5:, :]   # split first five data for after train test
after_train_test = test[:5, :]  # get first five data for after train test

# Step 5    lets load the data to our Module
jarvis_jr.load_data_set(data_set, 0.8, 0)   # parameters(<data>,<percentage_of_train_data>,<output_column_index>)

# Step 6    setup the regression type, in our case it's logistic regression
jarvis_jr.set_regression(Regressions.LogisticRegression)

# Step 7    lets train our module
jarvis_jr.start_learning()

# Step 8    if we do not call this function our data will be lost after execution
jarvis_jr.save_trained_module("jarvis_jr")  # parameters(<module_name>) saves trained module as .txt file

# Step 9    lets load our trained module to try
"""
PS: You can run the first 8 steps just once and than close the steps: 5,6,7
    After that try to run steps: 1,2,3,4,9,10, because you do not have to 
    train your module again if you are gonna use the same module.
"""
jarvis_jr.load_trained_module("jarvis_jr")  # parameters(<module_name>) must be same with saved module name

# Step 10   after loading the trained module lets give an input and wait for guesses
jarvis_jr.test_given_input(after_train_test)

"""
    Remainder: CustomAI.py is currently just a prototype and might contain some bugs.
    So do not expect a perfectly running AI, feel free to give advise, bug reports and questions.
"""
