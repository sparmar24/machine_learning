""" Natural Language Processing for restaurent reviews (sentiment analysis) """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk  # used to ignore the words not useful i.e., stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
from config import inputfile


def pred_score(ps_estimator, ps_train_X, ps_test_X, ps_train_y, ps_test_y):
    """predict score for each ML model"""
    ps_estimator.fit(ps_train_X, ps_train_y)
    prediction = ps_estimator.predict(ps_test_X)
    confusion = confusion_matrix(ps_test_y, prediction)
    accuracy = accuracy_score(ps_test_y, prediction)
    precision = precision_score(ps_test_y, prediction, average="binary")
    recall = recall_score(ps_test_y, prediction, average="binary")
    f1score = f1_score(ps_test_y, prediction, average="binary")
    return confusion, accuracy, precision, recall, f1score


def main():
    dataset = pd.read_csv(inputfile)
    # dataset = pd.read_csv(
    #     "~/PycharmProjects/MLProjects_UdemyCourse/part7_NaturalLanguageProcessing/Restaurant_Reviews.tsv",
    #     delimiter="\t",
    #     quoting=3,
    # )

    """Data cleaning"""
    # import modules for deep cleaning the texts in dataset
    nltk.download("stopwords")
    # removes all conjugations to make the reviews simple e.g loved to love

    # create an empty list
    corpus = []
    # for loop to clean all the 1000 reviews
    for i in range(0, 1000):
        review = re.sub(
            "[^a-zA-Z]", " ", dataset.iloc[:, 0][i]
        )  # replace non letter by space
        # review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
        review = review.lower()  # all letters to lower case
        review = review.split()  # split the reviews in different columns

        # apply stemming now
        ps = PorterStemmer()
        all_stopwards = stopwords.words("english")
        all_stopwards.remove("not")
        review = [
            ps.stem(word) for word in review if not word in set(all_stopwards)
        ]
        review = " ".join(review)
        corpus.append(review)
        # print(corpus)

    """ Creating the Bag of Words model"""
    # tokenization process using scikit learn
    cv = CountVectorizer(max_features=1566)
    X = cv.fit_transform(
        corpus
    ).toarray()  # put all the words in reviews into different columns
    y = dataset.iloc[:, -1].values
    # print(len(X[0]))

    """ splitting the dataset into train and test data """
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    models_and_predictions = {}
    estimators = {
        "logistic": LogisticRegression(random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svc_linear": SVC(kernel="linear", random_state=0),
        "svc_rbf": SVC(kernel="rbf", random_state=0),
        "gaussian_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(
            criterion="entropy", random_state=0
        ),
        "random_forest": RandomForestClassifier(
            criterion="entropy", n_estimators=5, random_state=0
        ),
        "xgboost": XGBClassifier(),
    }

    for estimator_name, estimator in estimators.items():
        confusion, accuracy, precision, recall, f1score = pred_score(
            estimator, train_X, test_X, train_y, test_y
        )

        models_and_predictions[f"{estimator_name}"] = {
            "confusion": confusion,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1score": f1score,
        }


if __name__ == "__main__":
    main()

"""
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
print(new_y_pred)"""
