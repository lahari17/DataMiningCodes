#-------------------------------------------------------------------------
# AUTHOR: Lahari
# FILENAME: bagging_random_forest.py
# SPECIFICATION: comparing ensemble methods
# FOR: CS 5990- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = [0,0,0,0,0,0,0,0,0,0]  #this array will be used to count the votes of each classifier

#reading the training data from a csv file and populate dbTraining
#--> add your Python code here
dbTraining = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library
#reading the test data from a csv file and populate dbTest
#--> add your Python code here

#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here
dbTest = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library
total_test_samples = len(dbTest.index)
X_test = np.array(dbTest.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(dbTest.values)[:,-1]
print("Started my base and ensemble classifier ...")

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

  bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

  #populate the values of X_training and y_training by using the bootstrapSample
  #--> add your Python code here
  X_training = np.array(bootstrapSample.values)[:,:64]
  y_training = np.array(bootstrapSample.values)[:,-1]


  #fitting the decision tree to the data
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
  clf = clf.fit(X_training, y_training)
  numacc=0

  for i, testSample in enumerate(dbTest):

      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      #--> add your Python code here
      prediction = int(clf.predict([dbTest.iloc[i,:64]])[0])
      true_label = dbTest.iloc[i,-1]
      classVotes[prediction] += 1
      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
          #--> add your Python code here
          if prediction == true_label:
              numacc += 1

  if k == 0: #for only the first base classifier, print its accuracy here
     #--> add your Python code here
     accuracy = numacc / total_test_samples
     print("Finished my base classifier (fast but relatively low accuracy) ...")
     print("My base classifier accuracy: " + str(accuracy))
     print("")

  #now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
  #--> add your Python code here
trulabel_counts = [0,0,0,0,0,0,0,0,0,0]
trulabels = dbTest.iloc[:,-1:]
for index in range(len(trulabels)):
    truth = int(trulabels.iloc[index])
    trulabel_counts[truth] += 1

errors = 0
for i in range(10):
    errors += abs(classVotes[i] - trulabel_counts[i])

accuracy = 1 - (errors / total_test_samples)

#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
#--> add your Python code here
accurate = 0
for x, y in zip(X_test, y_test):
    predicted = clf.predict([x])
#compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
#--> add your Python code here
    if predicted[0] == y:
        accurate += 1

accuracy = accurate / total_test_samples
#printing Random Forest accuracy here
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")

# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# importing some Python libraries
# from sklearn import tree
# from sklearn.utils import resample
# from sklearn.ensemble import RandomForestClassifier
# import csv

# dbTraining = []
# dbTest = []
# X_training = []
# y_training = []
# classVotes = [] #this array will be used to count the votes of each classifier

# #reading the training data from a csv file and populate dbTraining
# with open('optdigits.tra', 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         dbTraining.append(row)

# #reading the test data from a csv file and populate dbTest
# with open('optdigits.tes', 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         dbTest.append(row)

# #inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
# for i in range(len(dbTest)):
#     classVotes.append([0,0,0,0,0,0,0,0,0,0])

# print("Started my base and ensemble classifier ...")

# accuracy = 0  # Initialize accuracy variable

# for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

#     bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)
    
#      # Reset variables for the current iteration
#     X_training.clear()
#     y_training.clear()
#     #populate the values of X_training and y_training by using the bootstrapSample
#     for item in bootstrapSample:
#         X_training.append([float(i) for i in item[:-1]]) # features
#         y_training.append(item[-1]) # class 
        
#         print("HII",X_training)
#         print("Hello",y_training) 


#     #fitting the decision tree to the data
#     clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
#     clf = clf.fit(X_training, y_training)

#     for i, testSample in enumerate(dbTest):

#         #make the classifier prediction for each test sample and update the corresponding index value in classVotes.
#         prediction = clf.predict([list(map(float, testSample[:-1]))])[0]
#         classVotes[i][int(prediction)] += 1

#         if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
#             if prediction == testSample[-1]:
#                 accuracy += 1

#     if k == 0: #for only the first base classifier, print its accuracy here
#         accuracy = accuracy/len(dbTest) * 100
#         print("Finished my base classifier (fast but relatively low accuracy) ...")
#         print("My base classifier accuracy: " + str(accuracy) + "%")
#         print("")
        
#     # Reset variables for the next iteration
#     X_training.clear()
#     y_training.clear() 

    

# # Calculate ensemble accuracy
# accuracy = 0
# for i, testSample in enumerate(dbTest):
#     prediction = classVotes[i].index(max(classVotes[i]))
#     if prediction == testSample[-1]:
#         accuracy += 1

# accuracy = accuracy/len(dbTest) * 100
# print("Finished my ensemble classifier (slow but higher accuracy) ...")
# print("My ensemble accuracy: " + str(accuracy) + "%")
# print("")

# # Reset variables for next iteration
# X_training.clear()
# y_training.clear()

# print("Started Random Forest algorithm ...")

# # Create a Random Forest Classifier
# clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

# # Fit Random Forest to the training data
# clf.fit(X_training,y_training)

# # Make the Random Forest prediction for each test sample.
# accuracy = 0
# for i, testSample in enumerate(dbTest):
#     prediction = clf.predict([list(map(float, testSample[:-1]))])[0]
#     if prediction == testSample[-1]:
#         accuracy += 1

# # Calculate Random Forest accuracy
# accuracy = accuracy/len(dbTest) * 100
# print("Random Forest accuracy: " + str(accuracy) + "%")

# print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
# # #importing some Python libraries
# # from sklearn import tree
# # from sklearn.utils import resample
# # from sklearn.ensemble import RandomForestClassifier
# # import csv

# # dbTraining = []
# # dbTest = []
# # X_training = []
# # y_training = []
# # classVotes = [] #this array will be used to count the votes of each classifier

# # #reading the training data from a csv file and populate dbTraining
# # with open('optdigits.tra', 'r') as csvfile:
# #     reader = csv.reader(csvfile)
# #     for row in reader:
# #         dbTraining.append(row)

# # #reading the test data from a csv file and populate dbTest
# # with open('optdigits.tes', 'r') as csvfile:
# #     reader = csv.reader(csvfile)
# #     for row in reader:
# #         dbTest.append(row)

# # print("Length of training data:", len(dbTraining))
# # print("Length of test data:", len(dbTest))

# # #inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
# # for i in range(len(dbTest)):
# #     classVotes.append([0,0,0,0,0,0,0,0,0,0])

# # print("Started my base and ensemble classifier ...")

# # accuracy = 0  # Initialize accuracy variable

# # for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

# #     bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)
    
# #     # Reset variables for the current iteration
# #     X_training.clear()
# #     y_training.clear()
# #     #populate the values of X_training and y_training by using the bootstrapSample
# #     for item in bootstrapSample:
# #         X_training.append([float(i) for i in item[:-1]]) # features
# #         y_training.append(item[-1]) # class 

        
# #     print("Length of X_training:", len(X_training))
# #     print("Length of y_training:", len(y_training))

# #     #fitting the decision tree to the data
# #     clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
# #     clf = clf.fit(X_training, y_training)

# #     for i, testSample in enumerate(dbTest):

# #         #make the classifier prediction for each test sample and update the corresponding index value in classVotes.
# #         prediction = clf.predict([list(map(float, testSample[:-1]))])[0]
# #         classVotes[i][int(prediction)] += 1


# #         if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
# #             if prediction == testSample[-1]:
# #                 accuracy += 1

# #     print("prediction",prediction)
# #     print("classvotes",classVotes)
    

# #     if k == 0: #for only the first base classifier, print its accuracy here
# #         accuracy = accuracy/len(dbTest) * 100
# #         print("Finished my base classifier (fast but relatively low accuracy) ...")
# #         print("My base classifier accuracy: " + str(accuracy) + "%")
# #         print("")
        
# #     # Reset variables for the next iteration
# #     X_training.clear()
# #     y_training.clear() 

# # # Calculate ensemble accuracy
# # accuracy = 0
# # for i, testSample in enumerate(dbTest):
# #     prediction = classVotes[i].index(max(classVotes[i]))
# #     if prediction == testSample[-1]:
# #         accuracy += 1

# # accuracy = accuracy/len(dbTest) * 100
# # print("Finished my ensemble classifier (slow but higher accuracy) ...")
# # print("My ensemble accuracy: " + str(accuracy) + "%")
# # print("")

# # # Reset variables for next iteration
# # X_training.clear()
# # y_training.clear()

# # print("Started Random Forest algorithm ...")

# # # Create a Random Forest Classifier
# # clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

# # # Fit Random Forest to the training data
# # clf.fit(X_training,y_training)

# # # Make the Random Forest prediction for each test sample.
# # accuracy = 0
# # for i, testSample in enumerate(dbTest):
# #     prediction = clf.predict([list(map(float, testSample[:-1]))])[0]
# #     if prediction == testSample[-1]:
# #         accuracy += 1

# # # Calculate Random Forest accuracy
# # accuracy = accuracy/len(dbTest) * 100
# # print("Random Forest accuracy: " + str(accuracy) + "%")

# # print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
