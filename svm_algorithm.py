import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# Displaying labels
print(cancer.feature_names)

# Displaying targets
print(cancer.target_names)

X = cancer.data

# ['malignant' 'benign'] -> 0, 1
classes = ['malignant' 'benign']
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

print(x_train, y_train)

# Support Vector Classification
# When no parameters are used,
# the accuracy obtained will be just above the half,
# by using some parameters, we can optimize the accuracy
# Accuracy rises because of DIMENSION INTRODUCTION
# C value decides the number of points allow in the margin
# C = 0 is the hard margin
# C = 1 VS C = 2, double the number of points in the margin
clf = svm.SVC(kernel="linear", C=1)
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

# Compare the two targets
acc = metrics.accuracy_score(y_test, y_predict)

print("Accuracy : \n", acc)
