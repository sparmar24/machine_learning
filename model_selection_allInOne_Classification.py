##########################################################################
######## Classification Model Selection (python code) #########
##########################################################################

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split the dataset into train and test data
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

# perform feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X  = sc.transform(test_X)

# Implementing models one by one to compare the "Accuracy Score"
'''#### logistic regression ####'''
from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression(random_state = 0)
lreg.fit(train_X, train_y)
yl_pred = lreg.predict(test_X)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(test_y, yl_pred)
ac = accuracy_score(test_y, yl_pred)
print(cm)
print("Logistic accuracy score = " , ac)


'''#### KNN regression ####'''
from sklearn.neighbors import KNeighborsClassifier
knnreg = KNeighborsClassifier(n_neighbors=5)
knnreg.fit(train_X, train_y)
yknn_pred = knnreg.predict(test_X)

cm_knn = confusion_matrix(test_y, yknn_pred)
ac_knn = accuracy_score(test_y, yknn_pred)
print(cm_knn)
print("KNN accuracy score = " , ac_knn)


'''#### SVM linear regression ####'''
from sklearn.svm import SVC
svcreg = SVC(kernel = "linear", random_state=0)
svcreg.fit(train_X, train_y)
ysvc_pred = svcreg.predict(test_X)

cm_svc = confusion_matrix(test_y, ysvc_pred)
ac_svc = accuracy_score(test_y, ysvc_pred)
print(cm_svc)
print("SVM accuracy score = " , ac_svc)


'''#### Kernel svm regression ####'''
from sklearn.svm import SVC
svcreg_rbf = SVC(kernel = "rbf", random_state = 0)
svcreg_rbf.fit(train_X, train_y)
yrbf_pred = svcreg_rbf.predict(test_X)

cm_rbf = confusion_matrix(test_y, yrbf_pred)
ac_rbf = accuracy_score(test_y, yrbf_pred)
print(cm_rbf)
print("Kernel SVM accuracy score = " , ac_rbf)


'''#### Naive Bayes regression ####'''
from sklearn.naive_bayes import GaussianNB
bayes_reg = GaussianNB()
bayes_reg.fit(train_X, train_y)
ybayes_pred = bayes_reg.predict(test_X)

cm_b = confusion_matrix(test_y, ybayes_pred)
ac_b = accuracy_score(test_y, ybayes_pred)
print(cm_b)
print("Naive Bayes accuracy score = " , ac_b)


'''#### Decision Tree regression ####'''
from sklearn.tree import DecisionTreeClassifier
d_reg = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
d_reg.fit(train_X, train_y)
yd_pred = d_reg.predict(test_X)

cm_d = confusion_matrix(test_y, yd_pred)
ac_d = accuracy_score(test_y, yd_pred)
print(cm_d)
print("Decision Tree accuracy score = " , ac_d)


'''#### Random Forest regression ####'''
from sklearn.ensemble import RandomForestClassifier
rf_reg = RandomForestClassifier(criterion = "entropy", n_estimators = 5, random_state = 0)
rf_reg.fit(train_X, train_y)
yrf_pred = rf_reg.predict(test_X)

cm_rf = confusion_matrix(test_y, yrf_pred)
ac_rf = accuracy_score(test_y, yrf_pred)
print(cm_rf)
print("Random Forest accuracy score = " , ac_rf)




