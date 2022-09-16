######################################################################
## Breast cancer case study using different Machine Learning models ##
######### applied "k-fold cross" validation for best ML model ########
######################################################################

# "csv" data file from UCI machine learning repository :https://archive.ics.uci.edu/ml/index.php

import pandas as pd
import numpy as np

# load the dataset
dataset = pd.read_csv("Breast_Cancer_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split the training and test data
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25)

# data preprocessing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X) 
test_X  = sc.transform(test_X)

# training different models on training data
''' #### Logistic Regression ####'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(train_X, train_y)
y_pred = classifier.predict(test_X)

# calculating confusion matrix: {fp, fn, tp, tn} and accuracy score for each model one by one
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(test_y, y_pred)
ac = accuracy_score(test_y, y_pred)
print("Logistic Regression accuracy score = " , ac)

'''#### KNN regression ####'''
from sklearn.neighbors import KNeighborsClassifier
knnreg = KNeighborsClassifier(n_neighbors=5)
knnreg.fit(train_X, train_y)
yknn_pred = knnreg.predict(test_X)
#print(yknn_pred)

cm_knn = confusion_matrix(test_y, yknn_pred)
ac_knn = accuracy_score(test_y, yknn_pred)
print("KNN accuracy score = " , ac_knn)

'''#### svm linear regression ####'''
from sklearn.svm import SVC
svcreg = SVC(kernel = "linear", random_state=0)
svcreg.fit(train_X, train_y)
ysvc_pred = svcreg.predict(test_X)

cm_svc = confusion_matrix(test_y, ysvc_pred)
ac_svc = accuracy_score(test_y, ysvc_pred)
print("SVM accuracy score = " , ac_svc)

'''#### Kernel svm regression ####'''
from sklearn.svm import SVC
svcreg_rbf = SVC(kernel = "rbf", random_state = 0)
svcreg_rbf.fit(train_X, train_y)
yrbf_pred = svcreg_rbf.predict(test_X)

cm_rbf = confusion_matrix(test_y, yrbf_pred)
ac_rbf = accuracy_score(test_y, yrbf_pred)
print("Kernel SVM accuracy score = " , ac_rbf)

'''#### Naive Bayes regression ####'''
from sklearn.naive_bayes import GaussianNB
bayes_reg = GaussianNB()
bayes_reg.fit(train_X, train_y)
ybayes_pred = bayes_reg.predict(test_X)

cm_b = confusion_matrix(test_y, ybayes_pred)
ac_b = accuracy_score(test_y, ybayes_pred)
print("Naive Bayes accuracy score = " , ac_b)

'''#### Decision Tree regression ####'''
from sklearn.tree import DecisionTreeClassifier
d_reg = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
d_reg.fit(train_X, train_y)
yd_pred = d_reg.predict(test_X)

cm_d = confusion_matrix(test_y, yd_pred)
ac_d = accuracy_score(test_y, yd_pred)
print("Decision Tree accuracy score = " , ac_d)

'''#### Random Forest regression ####'''
from sklearn.ensemble import RandomForestClassifier
rf_reg = RandomForestClassifier(criterion = "entropy", n_estimators = 5, random_state = 0)
rf_reg.fit(train_X, train_y)
yrf_pred = rf_reg.predict(test_X)

cm_rf = confusion_matrix(test_y, yrf_pred)
ac_rf = accuracy_score(test_y, yrf_pred)
print("Random Forest accuracy score = " , ac_rf)
print("==============================================")


print("accuracy score after applying k-fold validation (here, k = 10)")
from sklearn.model_selection import cross_val_score
score1 = cross_val_score(estimator=classifier, X= train_X, y= train_y, cv = 10)
score2 = cross_val_score(estimator=knnreg, X= train_X, y= train_y, cv = 10)
score3 = cross_val_score(estimator=svcreg, X= train_X, y= train_y, cv = 10)
score4 = cross_val_score(estimator=svcreg_rbf, X= train_X, y= train_y, cv = 10)
score5 = cross_val_score(estimator=bayes_reg, X= train_X, y= train_y, cv = 10)
score6 = cross_val_score(estimator=d_reg, X= train_X, y= train_y, cv = 10)
score7 = cross_val_score(estimator=rf_reg, X= train_X, y= train_y, cv = 10)
# print(score1)
# print(score2)
# print(score3)
# print(score4)
# print(score5)
# print(score6)
# print(score7)

print("==============================================")
print("Logistic Regression acc_score: ", sum(score1)/len(score1))
print("kNN acc_score: ",sum(score2)/len(score2))
print("SVM acc_score: ", sum(score3)/len(score3))
print("SVM 'rbf' acc_score: ",sum(score4)/len(score4))
print("Naive Bayes acc_score: ",sum(score5)/len(score5))
print("Decision Tree acc_score: ",sum(score6)/len(score6))
print("Random Forest acc_score: ",sum(score7)/len(score7))
