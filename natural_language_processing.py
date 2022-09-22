# NLP model for sentiment analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import inputfile
dataset = pd.read_csv(inputfile)
   
'''Data cleaning'''
# import modules for deep cleaning the texts in dataset
import re
import nltk # used to ignore the words not useful, known as stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# removes all conjugations to make the reviews simple e.g loved to love

# create an empty list
corpus = []
#starting with for loop to clean all 1000 reviews
for i in range(0, 1000):
    review = re.sub("[^a-zA-Z]", " ", dataset.iloc[:, 0][i]) # replace non letter by space
    #OR review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower()  # all letters to lower case
    review = review.split()  # split the reviews in different columns

    # apply stemming now
    ps = PorterStemmer()
    all_stopwards = stopwords.words("english")
    all_stopwards.remove("not")
    review = [ps.stem(word) for word in review if not word in set(all_stopwards)]
    review = " ".join(review)
    corpus.append(review)
#print(corpus)

'''Creating the Bag of Words model'''
# tokenization process using scikit learn
from sklearn.feature_extraction.text import CountVectorizer
cv =  CountVectorizer(max_features = 1566)
X = cv.fit_transform(corpus).toarray()  # put all the words in reviews into different columns
y = dataset.iloc[:, -1].values
#print(len(X[0]))

'''splitting the dataset into train and test data sets'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)


'''Naive Bayes'''
print("Naive Bayes ---->")
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
score = accuracy_score(y_test, y_pred)
print("score: ", score)
pscore = precision_score(y_test, y_pred, average="binary")
print("pscore: ", pscore)
rscore = recall_score(y_test, y_pred, average="binary")
print("rscore: ", rscore)
f1score = f1_score(y_test, y_pred, average="binary")
print("f1score: ", f1score)

'''k-NN'''
print("Knn ---->")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
matrix_knn = confusion_matrix(y_test, y_pred_knn)
print(matrix_knn)
score_knn = accuracy_score(y_test, y_pred_knn)
print("score_knn: ", score_knn)
pscore_knn = precision_score(y_test, y_pred_knn, average="binary")
print("pscore_knn: ", pscore_knn)
rscore_knn = recall_score(y_test, y_pred_knn, average="binary")
print("rscore_knn: ", rscore_knn)
f1score_knn = f1_score(y_test, y_pred_knn, average="binary")
print("f1score_knn: ", f1score_knn)

'''SVM'''
print("SVM ---->")
from sklearn.svm import SVC
support = SVC(kernel="rbf", random_state=0)
support.fit(X_train, y_train)
y_pred_svm = support.predict(X_test)
matrix_svm = confusion_matrix(y_test, y_pred_svm)
print(matrix_svm)
score_svm = accuracy_score(y_test, y_pred_svm)
print("score_svm: ", score_svm)
pscore_svm = precision_score(y_test, y_pred_svm, average="binary")
print("pscore_svm: ", pscore_svm)
rscore_svm = recall_score(y_test, y_pred_svm, average="binary")
print("rscore_svm: ", rscore_svm)
f1score_svm = f1_score(y_test, y_pred_svm, average="binary")
print("f1score_svm: ", f1score_svm)


'''Random Forest'''
print("Random Forest ----> ")
from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=0)
random.fit(X_train, y_train)
y_pred_random = random.predict(X_test)
matrix_random = confusion_matrix(y_test, y_pred_random)
print(matrix_random)
score_random = accuracy_score(y_test, y_pred_random)
print("score_random: ", score_random)
pscore_random = precision_score(y_test, y_pred_random, average="binary")
print("pscore_random: ", pscore_random)
rscore_random = recall_score(y_test, y_pred_random, average="binary")
print("rscore_random: ", rscore_random)
f1score_random = f1_score(y_test, y_pred_random, average="binary")
print("f1score_random: ", f1score_random)


'''
 #Bonus class: Predicting a single review if positive or negative
textlist = []
#text = "I love this restaurent so much"
#text = "I do not love you"
#text = "I hate this restaurent so much"
#text = "I want to watch the movie"
text = "movie was not interesting"
text = re.sub("[^a-zA-Z]", " ", text)
text = text.lower()
text = text.split()

ps = PorterStemmer()
all_stopwards = stopwords.words("english")
all_stopwards.remove("not")
all_stopwards.remove("was")
text = [ps.stem(word) for word in text if not word in set(all_stopwards)]
text = " ".join(text)
textlist.append(text)
print(textlist)

new_X_test = cv.transform(textlist).toarray()
new_y_pred = random.predict(new_X_test) # can test the prediction with different models
print(new_y_pred)'''
