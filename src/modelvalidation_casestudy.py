""" Breast cancer case study using different Machine Learning models """
######### applied "k-fold cross" validation for best ML model estimation ########
# "csv" data file from UCI machine learning repository :https://archive.ics.uci.edu/ml/index.php

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def pred_score(ps_estimator, ps_train_X, ps_test_X, ps_train_y, ps_test_y):
    """predict accuracy for each model"""
    ps_estimator.fit(ps_train_X, ps_train_y)
    prediction = ps_estimator.predict(ps_test_X)
    confusion = confusion_matrix(ps_test_y, prediction)
    accuracy = accuracy_score(ps_test_y, prediction)
    return prediction, confusion, accuracy


def main():
    # load the dataset "Breast_Cancer_Data.csv"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    inputfile = f"{dir_path}/data/Breast_Cancer_Data.csv"
    dataset = pd.read_csv(inputfile)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    y = np.where(y > 2, 1, 0)

    # split the training and test data
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.25, random_state=40
    )

    # data preprocessing
    sc = StandardScaler()
    train_X = sc.fit_transform(train_X)
    test_X = sc.transform(test_X)

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
        prediction, confusion, accuracy = pred_score(
            estimator, train_X, test_X, train_y, test_y
        )
        # accuracy score after applying k-fold validation (here, k = 10)
        cross_validation = cross_val_score(
            estimator=estimator, X=train_X, y=train_y, cv=10
        )
        models_and_predictions[f"{estimator_name}"] = {
            "prediction": prediction,
            "confusion": confusion,
            "accuracy": accuracy,
            "avg_cvs": sum(cross_validation) / len(cross_validation),
        }


if __name__ == "__main__":
    main()
