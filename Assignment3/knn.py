# #-------------------------------------------------------------------------
# # AUTHOR: your name
# # FILENAME: title of the source file
# # SPECIFICATION: description of the program
# # FOR: CS 5990- Assignment #3
# # TIME SPENT: how long it took you to complete the assignment
# #-----------------------------------------------------------*/

# #importing some Python libraries
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier

# #11 classes after discretization
# classes = [i for i in range(-22, 40, 6)]

# #defining the hyperparameter values of KNN
# k_values = [i for i in range(1, 20)]
# p_values = [1, 2]
# w_values = ['uniform', 'distance']

# #reading the training data
# #reading the test data
# #hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')

# # reading the training data
# train_data = pd.read_csv('weather_training.csv')
# X_training = np.array(train_data.values[:, 1:-1]).astype('f')  # features
# y_training = np.array(train_data.values[:, -1]).astype('f')   # class labels

# # reading the test data
# test_data = pd.read_csv('weather_test.csv')
# X_test = np.array(test_data.values[:, 1:-1]).astype('f')  # features
# y_test = np.array(test_data.values[:, -1]).astype('f')   # class labels

# # Discretize the class values into 11 bins for training labels
# y_training_discretized = pd.qcut(y_training, 11, labels=False)
# # update the test class values according to the discretization (11 values only)
# y_test_discretized = pd.cut(y_test, bins=len(classes), labels=classes, include_lowest=True)
# #loop over the hyperparameter values (k, p, and w) ok KNN
# #--> add your Python code here
# for k in k_values:
#     for p in p_values:
#         for w in w_values:

#             #fitting the knn to the data
#             #--> add your Python code here

#             #fitting the knn to the data
#             clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
#             clf = clf.fit(X_training, y_training_discretized)

#             #make the KNN prediction for each test sample and start computing its accuracy
#             #hint: to iterate over two collections simultaneously, use zip()
#             #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
#             #to make a prediction do: clf.predict([x_testSample])
#             correct_predictions = 0
#             for x_testSample, y_testSample in zip(X_test, y_test_discretized):
#                 predicted_value = clf.predict([x_testSample])[0]
#                 real_value = y_testSample

#             #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
#             #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
#             #--> add your Python code here
#                 percentage_difference = 100 * abs(predicted_value - real_value) / real_value
#                 if percentage_difference <= 15:
#                     correct_predictions += 1
#             #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
#             #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
#             #--> add your Python code here

#             accuracy = correct_predictions / len(y_test)
#             highest_accuracy = 0 
#             if accuracy > highest_accuracy:
#                 highest_accuracy = accuracy
#                 best_params = (k, p, w)

#                 print("Highest KNN accuracy so far:", highest_accuracy, "Parameters:", "k =", k, ", p =", p, ", weight =", w)

#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: knn.py
# SPECIFICATION: This program reads the weather_training.csv (training set) and weather_test.csv (test set)
#                It discretizes the class values into 11 bins for both training and test sets.
#                Then, it performs a grid search to try multiple values for the KNN hyperparameters such as k
#                (number of neighbors), p (distance metric), and w (form of weights) value.
#                It updates and prints the highest accuracy calculated during the experiments.
#                Any predictions [-15%,+15%] distant from the real output values are considered correct.
# FOR: CS 5990- Assignment #3
# TIME SPENT: time taken to complete the assignment
#-----------------------------------------------------------*/

#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: knn.py
# SPECIFICATION: This program reads the weather_training.csv (training set) and weather_test.csv (test set).
#                It discretizes the class values into 11 bins for both training and test sets.
#                Then, it performs a grid search to try multiple values for the KNN hyperparameters (k, p, and w).
#                It updates and prints the highest accuracy calculated during the experiments.
#                Any predictions within [-15%,+15%] distant from the real output values are considered correct.
# FOR: CS 5990- Assignment #3
# TIME SPENT: time taken to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# 11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

# Discretization function
def discretize_temperature(value):
    for cl in classes:
        if value < cl + 3:
            return cl
    return classes[-1]

# Grid search parameters
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

# Reading and preprocessing data
training_df = pd.read_csv('weather_training.csv')
X_training = training_df.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_training = np.array([discretize_temperature(x) for x in training_df['Temperature (C)'].values]).astype(int)

test_df = pd.read_csv('weather_test.csv')
X_test = test_df.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_test_real = test_df['Temperature (C)'].values
y_test = np.array([discretize_temperature(x) for x in test_df['Temperature (C)'].values]).astype(int)

# Grid search and training
highest_accuracy = 0
best_params = {}

for k in k_values:
    for p in p_values:
        for w in w_values:
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf.fit(X_training, y_training)

            correct_predictions = 0
            for x_test_sample, y_test_real_value in zip(X_test, y_test_real):
                predicted_class = clf.predict([x_test_sample])[0]
                percentage_difference = 100 * abs(predicted_class - y_test_real_value) / abs(y_test_real_value)

                if percentage_difference <= 15:
                    correct_predictions += 1

            accuracy = correct_predictions / len(y_test_real)

            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_params = {'k': k, 'p': p, 'weight': w}
                print(f"Highest KNN accuracy so far: {highest_accuracy:.2f}")
                print(f"Parameters: k = {k}, p = {p}, weight = {w}")

# Output results
print(f"Best parameters: k = {best_params['k']}, p = {best_params['p']}, weight = {best_params['weight']}")
print(f"Highest KNN accuracy: {highest_accuracy:.2f}")
