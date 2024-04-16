#-------------------------------------------------------------------------
# AUTHOR: Lahari Sandepudi
# FILENAME: knn.py
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #3
# TIME SPENT: 2hrs
#-----------------------------------------------------------*/

# Importing necessary libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Define the discretized temperature classes
classes = [i for i in range(-22, 40, 6)]

# Function to discretize temperature
def discretize_temperature(value):
    for c in classes:
        if value <= c + 3:
            return c
    return classes[-1]

# Define the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

# Read the training data
training_data = pd.read_csv('weather_training.csv')
X_training = training_data.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_training = np.array([discretize_temperature(x) for x in training_data['Temperature (C)'].values])

# Read the test data
test_data = pd.read_csv('weather_test.csv')
X_test = test_data.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_test_real = test_data['Temperature (C)'].values
y_test = np.array([discretize_temperature(x) for x in test_data['Temperature (C)'].values])

# Loop over the hyperparameter values (k, p, and w) for KNN
highest_accuracy = 0
best_params = {}

for k in k_values:
    for p in p_values:
        for w in w_values:
            # Fitting the KNN to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf.fit(X_training, y_training)
            
            # Compute accuracy
            correct_predictions = 0
            for x_test_sample, y_test_real_value in zip(X_test, y_test_real):
                predicted_class = clf.predict([x_test_sample])[0]
                predicted_temperature = discretize_temperature(predicted_class)
                percentage_difference = 100 * abs(predicted_temperature - y_test_real_value) / abs(y_test_real_value)
                if percentage_difference <= 15:
                    correct_predictions += 1

            accuracy = correct_predictions / len(y_test_real)

            # Update highest accuracy if necessary
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_params = {'k': k, 'p': p, 'weight': w}
                print(f"Highest KNN accuracy so far: {highest_accuracy:.2f}")
                print(f"Parameters: k = {k}, p = {p}, weight = {w}")

# Output the best parameters and highest accuracy
print(f"Best parameters: k = {best_params['k']}, p = {best_params['p']}, weight = {best_params['weight']}")
print(f"Highest KNN accuracy: {highest_accuracy:.2f}")
