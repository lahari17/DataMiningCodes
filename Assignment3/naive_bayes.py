#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

classes = [i for i in range(-22, 40, 6)]

def discretize(instance):
    try:
        prev = -100
        for c in classes:
            if instance["Temperature (C)"] > prev and instance["Temperature (C)"] <= c:
                instance["Temperature (C)"] = c
            prev = c
    except:
        print("instance failed to discretize", instance)
    return instance

# 11 classes after discretization

# reading the training data
# --> add your Python code here
training_df = pd.read_csv('weather_training.csv')
training_df.dropna(how="all")
discrete_training_df = training_df.apply(discretize, axis=1)

# The columns that we will be making predictions with.
y_training = np.array(discrete_training_df["Temperature (C)"])
y_training = y_training.astype(dtype='int')
X_training = np.array(discrete_training_df.drop(["Temperature (C)", "Formatted Date"], axis=1).values)

# reading the test data
# --> add your Python code here
test_df = pd.read_csv('weather_test.csv')
test_df.dropna(how="all")

# update the test class values according to the discretization (11 values only)
discrete_test_df = test_df.apply(discretize, axis=1)
y_test = discrete_test_df["Temperature (C)"]
y_test = y_test.astype(dtype='int')
X_test = discrete_test_df.drop(["Temperature (C)", "Formatted Date"], axis=1).values

# fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, y_training)

# make the naive_bayes prediction for each test sample and start computing its accuracy
# the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
# to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
num_of_accurate = 0
for (x_testSample, y_testSample) in zip(X_test, y_test):
    prediction = clf.predict(np.array([x_testSample]))
    # the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
    # to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    diff = 100 * (abs(prediction[0] - y_testSample) / y_testSample)
    if diff >= -15 and diff <= 15:
        num_of_accurate += 1

# print the naive_bayes accuracy
# --> add your Python code here
score = num_of_accurate / len(y_test)
print(f"naive_bayes accuracy: {score}")

# from sklearn.naive_bayes import GaussianNB
# import pandas as pd
# import numpy as np
# #11 classes after discretization
# classes = [i for i in range(-22, 40, 6)]

# #reading the training data
# # training_path = r"Assignment3\weather_training.csv"
# data_training = pd.read_csv('weather_training.csv',sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)


# #update the training class values according to the discretization (11 values only)
# def discretize(instance):
#     try:
#         prev=-100
#         for c in classes:
#             if prev < instance["Temperature (C)"] <= c:
#                 instance["Temperature (C)"] = c
#             prev = c
#         if instance["Temperature (C)"] > c:
#             instance["Temperature (C)"] = c
#     except:
#         print("instance failed to discretize", instance)
#     return instance

# discreted_training_data = data_training.apply(discretize, axis = 1)

# # The columns that we will be making predictions with.
# y_training = np.array(discreted_training_data["Temperature (C)"]).astype(dtype='int')
# X_training = np.array(discreted_training_data.drop(["Temperature (C)","Formatted Date"], axis=1).values)


# #reading the test data
# # testing_path = r"Assignment3\weather_test.csv"
# # test_df = pd.read_csv(testing_path, sep=',', header=0)
# test_df = pd.read_csv('weather_test.csv',sep=',', header=0)

# #update the test class values according to the discretization (11 values only)
# discreted_testing_data = test_df.apply(discretize, axis=1)
# y_test = discreted_testing_data["Temperature (C)"].astype(dtype='int')
# X_test = discreted_testing_data.drop(["Temperature (C)","Formatted Date"], axis=1).values

# #fitting the naive_bayes to the data
# clf = GaussianNB()
# clf = clf.fit(X_training, y_training)

# #make the naive_bayes prediction for each test sample and start computing its accuracy
# #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
# #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
# accurates = 0
# for test_x , test_y in zip(X_test,y_test):
#     preditcted = clf.predict(np.array([test_x]))

#     # given that the prediction should be considered correct if the output value is [-15%,+15%] from the real output values.
#     approx = (abs(preditcted[0] - test_y)/test_y)*100
#     if -15<=approx<=15:
#         accurates+=1

# #print the naive_bayes accuracyy
# accuracy = accurates/len(y_test)
# print("naive_bayes accuracy: " + str(accuracy))
