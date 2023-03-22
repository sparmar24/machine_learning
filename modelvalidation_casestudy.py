## Breast cancer case study using different Machine Learning models ##
######### applied "k-fold cross" validation for best ML model ########
# "csv" data file from UCI machine learning repository :https://archive.ics.uci.edu/ml/index.php

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

# load the dataset
dataset = pd.read_csv("Breast_Cancer_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


def pred_score(ps_estimator, ps_train_X, ps_test_X, ps_train_y, ps_test_y):
    ps_estimator.fit(ps_train_X, ps_train_y)
    prediction = ps_estimator.predict(ps_test_X)
    # calculating confusion matrix: {fp, fn, tp, tn} and accuracy score for each model one by one
    confusion = confusion_matrix(ps_test_y, prediction)
    accuracy = accuracy_score(ps_test_y, prediction)
    return prediction, confusion, accuracy


def main():
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
    }

    for estimator_name, estimator in estimators.items():
        prediction, confusion, accuracy = pred_score(
            estimator, train_X, test_X, train_y, test_y
        )
        value = {"prediction": prediction, "confusion": confusion, "accuracy": accuracy}
        models_and_predictions[f"{estimator_name}"] = value

    breakpoint()

    # training different models on training data
    """ Logistic Regression """
    classifier = LogisticRegression(random_state=0)
    lr_predicted, lr_confusion, lr_accuracy = pred_score(
        classifier, train_X, test_X, train_y, test_y
    )
    """ KNN regression """
    knnreg = KNeighborsClassifier(n_neighbors=5)
    knn_predicted, knn_confusion, knn_accuracy = pred_score(
        knnreg, train_X, test_X, train_y, test_y
    )
    """ svm linear regression """
    svcreg = SVC(kernel="linear", random_state=0)
    svc_predicted, svc_confusion, svc_accuracy = pred_score(
        svcreg, train_X, test_X, train_y, test_y
    )
    """ Kernel svm regression """
    svcreg_rbf = SVC(kernel="rbf", random_state=0)
    rbf_predicted, rbf_confusion, rbf_accuracy = pred_score(
        svcreg_rbf, train_X, test_X, train_y, test_y
    )
    """ Naive Bayes regression """
    bayes_reg = GaussianNB()
    b_predicted, b_confusion, b_accuracy = pred_score(
        bayes_reg, train_X, test_X, train_y, test_y
    )
    """ Decision Tree regression """
    d_reg = DecisionTreeClassifier(criterion="entropy", random_state=0)
    d_predicted, d_confusion, d_accuracy = pred_score(
        d_reg, train_X, test_X, train_y, test_y
    )
    """ Random Forest regression """
    rf_reg = RandomForestClassifier(
        criterion="entropy", n_estimators=5, random_state=0
    )
    rf_predicted, rf_confusion, rf_accuracy = pred_score(
        rf_reg, train_X, test_X, train_y, test_y
    )

    print("==============================================")
    breakpoint()

    print("accuracy score after applying k-fold validation (here, k = 10)")
    score1 = cross_val_score(estimator=classifier, X=train_X, y=train_y, cv=10)
    score2 = cross_val_score(estimator=knnreg, X=train_X, y=train_y, cv=10)
    score3 = cross_val_score(estimator=svcreg, X=train_X, y=train_y, cv=10)
    score4 = cross_val_score(estimator=svcreg_rbf, X=train_X, y=train_y, cv=10)
    score5 = cross_val_score(estimator=bayes_reg, X=train_X, y=train_y, cv=10)
    score6 = cross_val_score(estimator=d_reg, X=train_X, y=train_y, cv=10)
    score7 = cross_val_score(estimator=rf_reg, X=train_X, y=train_y, cv=10)
    print("==============================================")
    # print("Logistic Regression acc_score: ", sum(score1) / len(score1))
    # print("kNN acc_score: ", sum(score2) / len(score2))
    # print("SVM acc_score: ", sum(score3) / len(score3))
    # print("SVM 'rbf' acc_score: ", sum(score4) / len(score4))
    # print("Naive Bayes acc_score: ", sum(score5) / len(score5))
    # print("Decision Tree acc_score: ", sum(score6) / len(score6))
    # print("Random Forest acc_score: ", sum(score7) / len(score7))


if __name__ == "__main__":
    main()
