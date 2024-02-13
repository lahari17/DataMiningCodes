# -------------------------------------------------------------------------
# AUTHOR: Lahari Sandepudi
# FILENAME: similarity.py
# SPECIFICATION: Python program  that will output the two most similar documents according to their cosine similarity.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 20min
# -----------------------------------------------------------*/

# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

# Use the following words as terms to create your document-term matrix
# [soccer, favorite, sport, like, one, support, olympic, games]
# --> Add your Python code here

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
# --> Add your Python code here


# Print the highest cosine similarity following the information below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
# --> Add your Python code here
words=["soccer", "favorite", "sport", "like", "one", "support", "olympic", "games"]
docarr1=[]
docstr=["doc1","doc2","doc3","doc4"]
docarr2=[doc1,doc2,doc3,doc4]


def docmatrix(doc):
    lst=doc.split()
    docarr=[]
    for i in words:
        c=0
        for j in lst:
            if i==j.replace(",",""):
                c=c+1
        docarr.append(c)
    docarr1.append(docarr)

for i in docarr2:
    docmatrix(i)

max=0
cs=cosine_similarity(docarr1)
for i in range(4):
    for j in range(4):
        if i!=j and cs[i][j]>max:
            max=cs[i][j]
            p=i
            q=j
print("The most similar documents are:" + str(docstr[p])+ " and " + str(docstr[q]) +" with cosine similarity = "+ str(max))









