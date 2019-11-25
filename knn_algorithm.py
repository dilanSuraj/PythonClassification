import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

# Converting non-numerical to numerical
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
class_ = le.fit_transform(list(data["class"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(class_)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

predictions = model.predict(x_test)

# Index positions are taken to print the name
names = ["unacc", "acc", "good", "vgood"]

print("Accuracy : \n", acc)

for x in range(len(predictions)):
    print("Prediction : ", names[predictions[x]], " Data : ", x_test[x], " Actual : ", names[y_test[x]])
    # distances between the neighbours and their indexes
    # array([[1., 1., 1., 1., 1., 1., 1.]]) <- distances
    # array([[ 332, 1098,  304,  210,   78,  502,  547]] <- indexes
    n = model.kneighbors([x_test[x]], 7, True)
    print("N : ", n)

