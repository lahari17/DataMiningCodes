<<<<<<< HEAD
# -------------------------------------------------------------------------
# AUTHOR: Lahari Sandepudi
# FILENAME: decision_tree.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: ~1 hr
# -----------------------------------------------------------*/
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define dictionaries for encoding categorical variables
marital_status = {"Single": [1, 0, 0], "Married": [0, 1, 0], "Divorced": [0, 0, 1]}
refund = {"Yes": 1, "No": 0}

# Function to process instance row for data transformation
def process_instance_row(instance):
    new_instance = []
    new_instance.append(refund[instance[0]])
    hot_encode = marital_status[instance[1]]
    new_instance.extend(hot_encode)
    taxable = float(instance[2].replace('k', ''))
    new_instance.append(taxable)
    return new_instance

# Define datasets
dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

# Function to calculate accuracy
def calculate_accuracy(clf, data_test, refund):
    tp = tn = fp = fn = 0
    for data in data_test:
        # Transform features of test instances and make class prediction
        transformed_test_instance = process_instance_row(data)
        class_predicted = clf.predict([transformed_test_instance])[0]

        # Compare prediction with the true label to calculate model accuracy
        test_class_value = refund[data[3]]
        if class_predicted == 1 and test_class_value == 1:
            tp += 1
        elif class_predicted == 1 and test_class_value == 0:
            fp += 1
        elif class_predicted == 0 and test_class_value == 1:
            fn += 1
        else:
            tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy

# Loop through datasets
for ds in dataSets:
    X = []
    Y = []

    # Read and process training data
    df = pd.read_csv('DataMiningCodes/Assignment2/'+ds, sep=',', header=0)
    data_training = np.array(df.values)[:, 1:]

    # Transform training data
    for instance in data_training:
        new_instance = process_instance_row(instance)
        X.append(new_instance)
        Y.append(refund[instance[3]])

    accuracies = []
    for i in range(10):
        # Fitting the decision tree to the data using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        # Plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'],
                       class_names=['Yes', 'No'], filled=True, rounded=True)
        plt.show()

        # Read the test data and add it to data_test
        cheat_test = pd.read_csv('DataMiningCodes\Assignment2\cheat_test.csv', sep=',', header=0)
        data_test = np.array(cheat_test.values)[:, 1:]

        # Calculate accuracy
        accuracy = calculate_accuracy(clf, data_test, refund)
        accuracies.append(accuracy)

    # Calculate average accuracy
    avg_accuracy = np.average(accuracies)

    # Print the accuracy of this model during the 10 runs
    print("Average accuracy when training on", ds, ":", avg_accuracy)
