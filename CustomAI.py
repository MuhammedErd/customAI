import numpy as np
from math import pow, e, log
from Enumerations import Algorithms, Regressions


# ------------------------------------------ #
# -----------  PRIVATE GLOBALS   ----------- #
# ------------------------------------------ #

#    Matrix Variables
__x = []                                # Param matrix
__y = []                                # Result matrix
__Q = []                                # Weight matrix
__test = []                             # Test matrix
__test_output = []                      # Test output matrix

#    Single Value Variables
__n = 0                                 # Parameter count
__m = 0                                 # Data set size
__alpha = 0.000001                      # Default learning rate

#    Optimization Variables
__algorithm_type = Algorithms.unused    # Default algorithm usage
__mean_normalization = False            # False by default
__norm_matrix = []                      # Np matrix for mean optimization
__learning_rate_optimization = False    # False by default

#    Module Helpers
__module_trained = False                # False by default
__regression_type = Regressions.unused  # Default regression usage
__module_name = "j.doe"                 # By default

# ---------------    END    ---------------- #


# ------------------------------------------ #
# -----------   USER FUNCTIONS   ----------- #
# ------------------------------------------ #


#   matrix, train data percentage
def load_data_set(data_set, data_to_train=0.8, output_index=0):
    if data_to_train >= 1.0 or data_to_train <= 0.0:
        data_to_train = 0.8     # if percentage is not valid fix it to default
        __ai_logger("Given percentage is not valid so that i fixed to 0.8(train) - 0.2(test)")
    global __x, __y, __n, __m, __Q, __test, __test_output
    data_set = __handle_structured_data(data_set, output_index)
    train_data = __extract_train_data(data_set, data_to_train)
    test_data = __extract_test_data(data_set, (1-data_to_train))

    __x = __get_train_data_input(train_data)
    __y = __get_train_data_output(train_data)

    __test = __get_test_data_input(test_data)
    __test_output = __get_test_data_output(test_data)

    __m = __x.shape[1]
    __n = __x.shape[0]-1
    __Q = np.zeros((__n+1, 1))


#   set Regression type
def set_regression(regression=Regressions.unused):
    global __regression_type, __mean_normalization, __learning_rate_optimization, __algorithm_type
    if regression is Regressions.LinearRegression:
        __regression_type = Regressions.LinearRegression
        # Default optimizations for selected regression
        __mean_normalization = True
        __learning_rate_optimization = True
        __algorithm_type = Algorithms.GradientDescent
        return True
    elif regression is Regressions.LogisticRegression:
        __regression_type = Regressions.LogisticRegression
        # Default optimization for selected regression
        __learning_rate_optimization = True
        __algorithm_type = Algorithms.GradientDescent
        return True
    return False


"""
#   set optimizations
def set_optimizations():
"""


#   starts learning process
def start_learning():
    # check if there is 'before learning' optimization
    if __mean_normalization is True:
        __mean_op()
    if __algorithm_type is Algorithms.GradientDescent:
        __gradient_descent()
    else:
        __ai_logger("Failed to call algorithm")

    global __module_trained
    __module_trained = True  # set module as "trained"
    # call test
    __test_case()


#  save trained module
def save_trained_module(name=__module_name):
    global __module_name
    if __module_trained is False:
        __ai_logger("You must train me first!")
        return False
    __module_name = name
    foo = open(name+".txt", "w")
    foo.write(str(__Q.shape[0]))
    for i in range(__Q.shape[0]):
        foo.write("#"+str(__Q[i]))
    if __regression_type is Regressions.LogisticRegression:
        module_regression = "\nLogisticRegression"
    foo.write(module_regression)    # save regression
    foo.close()
    __ai_logger("Module is saved.")


#  load trained module
def load_trained_module(name=__module_name):
    try:
        foo = open(name+".txt", "r")
    except FileNotFoundError:
        __ai_logger("Given module name is not found!")
        return False
    file_content = foo.readline().split('#')
    regression = foo.readline()
    foo.close()
    param_count = int(file_content[0])
    if param_count != (len(file_content)-1):
        __ai_logger("Given module is incorrect!")
        return False
    global __Q, __module_trained, __regression_type
    if regression == "LogisticRegression":
        __regression_type = Regressions.LogisticRegression
    elif regression == "LinearRegression":
        __regression_type = Regressions.LinearRegression
    else:
        __ai_logger("Given module is incorrect!")
        return False
    temp_q = np.zeros((param_count, 1))

    for i in range(1, param_count+1):
        if i == param_count:
            value = file_content[i][1:-2]
        else:
            value = file_content[i][1:-1]
        temp_q[i-1] = float(value)

    __Q = temp_q
    __ai_logger("Module is loaded successfully.")
    __module_trained = True
    return True


#   try test case
def test_given_input(data, output_column_index=0):    # by default
    if __module_trained is False:
        __ai_logger("You must train me first!")
        return False
    data = __handle_structured_data(data, output_column_index)
    data = __extract_test_data(data, 1)
    global __test, __test_output
    __test = __get_test_data_input(data)
    __test_output = __get_test_data_output(data)
    # execute the test
    __test_case()
    return True


# ------------------------------------------ #
# -----------     ALGORITHMS     ----------- #
# ------------------------------------------ #


# Gradient Descent
def __gradient_descent():
    global __x, __y, __Q, __m, __n, __alpha
    q_temp = np.zeros((__n+1, 1))
    if __learning_rate_optimization is True:
        rate = __alpha/20
    initial_cost = __cost_function()
    try_count = 0
    while True:
        for j in range(__n+1):
            q_temp[j] = (__alpha/__m)*__plus(j)
        __Q -= q_temp
        current_cost = __cost_function()
        if current_cost > initial_cost:
            print("exit status current cost:", current_cost, "initial cost: ", initial_cost)
            __Q += q_temp
            break
        elif __learning_rate_optimization is True and initial_cost-current_cost <= 0.1:
            __alpha += rate
        elif __learning_rate_optimization is True:
            __alpha = rate*20

        initial_cost = current_cost
        try_count += 1
        print("cost: ", current_cost)


# ---------------    END    ---------------- #


# ------------------------------------------ #
# -----------  PARENT FUNCTIONS  ----------- #
# ------------------------------------------ #


def __cost_function():
    if __regression_type is Regressions.LinearRegression:
        return __linear_regression_cost_function()
    elif __regression_type is Regressions.LogisticRegression:
        return __logistic_regression_cost_function()
    else:
        __ai_logger("FAILED TO CALL COST FUNCTION")

    return 0    # fail


def __plus(j):
    global __m, __x, __y
    result = 0
    for i in range(__m):
        result += ((__hypo(i)-__y[0][i])*__x[j][i])

    return result


def __hypo(i):
    if __regression_type is Regressions.LinearRegression:
        return __linear_regression_hypo(i)
    elif __regression_type is Regressions.LogisticRegression:
        return __logistic_regression_hypo(i)
    else:
        __ai_logger("FAILED TO CALL HYPO FUNCTION")
    return 0    # fail

# ---------------    END    ---------------- #


# ------------------------------------------ #
# -----------  CHILD FUNCTIONS   ----------- #
# ------------------------------------------ #


def __linear_regression_cost_function():
    result = 0
    global __y, __m
    for i in range(__m):
        y = __y[0][i]
        hypo = __hypo(i)
        result += pow(hypo-y, 2)
    result /= (2*__m)
    return result


def __linear_regression_hypo(i):
    global __x, __Q
    return np.dot(__Q.transpose(), __x[:, [i]])


def __logistic_regression_cost_function():    # cost function for logistic regression
    result = 0
    global __y, __m
    for i in range(__m):
        y = __y[0][i]
        hypo = __hypo(i)
        result += y*log(hypo, 10)+(1-y)*log((1-hypo), 10)
    result /= (-1*__m)
    return result


def __logistic_regression_hypo(i):
    global __x, __Q
    z = np.dot(__Q.transpose(), __x[:, [i]])
    r = __sigmoid_function(z)
    return r


def __sigmoid_function(z):
    result = 1/(1+pow(e, -z))
    if result == 0:         # Handle edges
        result = 0.0000001
    elif result == 1:
        result = 0.9999999
    return result

# ---------------    END    ---------------- #


# ------------------------------------------ #
# -----------   OPTIMIZATIONS    ----------- #
# ------------------------------------------ #


def __mean_op():
    global __x, __Q, __norm_matrix, __n
    __norm_matrix = np.zeros((__n, 2))   # except x0
    for i in range(__n):            # jump x0
        data_row = __x[[i+1], :]
        temp_avg = np.mean(data_row)
        temp_min = data_row.min()
        temp_max = data_row.max()
        __x[[i+1], :] -= temp_avg
        __x[[i+1], :] /= (temp_max - temp_min)
        __norm_matrix[i][0] = temp_avg
        __norm_matrix[i][1] = (temp_max-temp_min)


def __apply_mean_norm_op(matrix):
    global __norm_matrix, __n
    for i in range(__n):
        matrix[[i+1], :] -= __norm_matrix[i][0]
        matrix[[i+1], :] /= __norm_matrix[i][1]
    return matrix

"""
def __learning_rate_op(initial_cost, current_cost):
"""

# ---------------    END    ---------------- #


# pre-condition for usage: module must be trained
# ------------------------------------------ #
# ------- TRAINED MODULE FUNCTIONS  -------- #
# ------------------------------------------ #


def __test_case():
    if __module_trained is False:
        __ai_logger("You must train me first!")
        return False
    global __test, __test_output
    if __mean_normalization is True:
        __test = __apply_mean_norm_op(__test)
    # start test
    correct_answer_count = 0
    test_data_size = __test.shape[1]
    for i in range(test_data_size):
        print("assumed answer: ", __hypothesis(__test[:, [i]]), " real answer: ", __test_output[0][i])

    for i in range(test_data_size):
        if __hypothesis(__test[:, [i]]) >= 0.5:
            if __test_output[0][i] == 1:
                correct_answer_count += 1
        else:
            if __test_output[0][i] == 0:
                correct_answer_count += 1
    result_log = "Test result : %" + str((correct_answer_count*100)/test_data_size)\
                 + " of the test data is answered correctly."
    __ai_logger(result_log)


def __hypothesis(input_data):
    global __Q
    r = np.dot(__Q.transpose(), input_data)
    if __regression_type is Regressions.LogisticRegression:
        r = __sigmoid_function(r)
    return r

# ---------------    END    ---------------- #

# ------------------------------------------ #
# ---------   HELPER FUNCTIONS    ---------- #
# ------------------------------------------ #


def get_module_info():
    print("Module name: ", __module_name)
    print("Train data set:\n", __x)
    print("Train data output:\n", __y)
    print("Weights:\n", __Q)
    print("Number of parameters: ", __n)
    print("Train data set size: ", __m)
    print("Is module trained: ", __module_trained)
    print("Test data set:\n", __test)
    print("Test data output:\n", __test_output)
    if __regression_type == 0:
        print("Selected regression type: not selected")
    elif __regression_type == 1:
        print("Selected regression type: Linear Regression")
    elif __regression_type == 2:
        print("Selected regression type: Logistic Regression")
    if __algorithm_type == 0:
        print("Selected algorithm type: not selected")
    elif __algorithm_type == 1:
        print("Selected algorithm type: Gradient Descent")
    print("----OPTIMIZATIONS----")
    print("Mean optimization: ", __mean_normalization)
    print("Learning rate optimization: ", __learning_rate_optimization)


"""
    [ x0 x0 x0 x0 x0 x0 x0 x0 .... x0 ]
    [ x1 x1 x1 x1 x1 x1 x1 x1 .... x1 ]
    [ x2 x2 x2 x2 x2 x2 x2 x2 .... x2 ]
    [ .. .. .. .. .. .. .. .. .... .. ]
    [ xn xn xn xn xn xn xn xn .... xn ]
    [ y  y  y  y  y  y  y  y  .... y  ] (n+2) x ( m ) matrix 
"""


# takes matrix and fix the data
def __handle_structured_data(data, output_data_clm_index):
    arr = np.array(data)
    if output_data_clm_index < 0 or output_data_clm_index >= arr.shape[1]:
        # type an error here
        return False
    # check for missing values here
    first_part = arr[:, :output_data_clm_index]
    output_part = arr[:, [output_data_clm_index]]
    second_part = arr[:, output_data_clm_index+1:]
    second_part = np.append(second_part, output_part, axis=1)
    arr = np.append(first_part, second_part, axis=1)
    arr = arr.transpose()
    return arr


# takes data set and returns train matrix
def __extract_train_data(date_set, percentage=0.8):
    index = int(date_set.shape[1]*percentage)
    train_data = date_set[:, :index]
    ones = np.ones((1, train_data.shape[1]))
    return np.append(ones, train_data, axis=0)


# takes data sen and returns test matrix
def __extract_test_data(data_set, percentage=0.2):
    index = int(data_set.shape[1]-data_set.shape[1]*percentage)
    test_data = data_set[:, index:]
    ones = np.ones((1, test_data.shape[1]))
    return np.append(ones, test_data, axis=0)


# takes train data and extract input
def __get_train_data_input(matrix):
    train_input = matrix[:matrix.shape[0]-1, :]
    return np.array(train_input, dtype=float)


# takes train data and extract output
def __get_train_data_output(matrix):
    train_output = matrix[[matrix.shape[0]-1], :]
    return np.array(train_output, dtype=float)


# takes test data and extract input
def __get_test_data_input(matrix):
    test_input = matrix[:matrix.shape[0]-1, :]
    return np.array(test_input, dtype=float)


# takes test data and extract output
def __get_test_data_output(matrix):
    test_output = matrix[[matrix.shape[0]-1], :]
    return np.array(test_output, dtype=float)


# AI LOGGER
def __ai_logger(given_log):
    print("[" + __module_name + "] " + given_log)
