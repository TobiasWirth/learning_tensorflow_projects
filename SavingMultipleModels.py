import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot

from matplotlib import style

# import data
data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1)) # All Features, drop G3 column from table, training data
y = np.array(data[predict]) # All labels, only G3 column

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

best_score = 0

"""
for _ in range (30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()
    
    linear.fit(x_train, y_train)
    
    accuracy = linear.score(x_test, y_test)
    print("Accuracy: \n", accuracy)

    # save the best model (highest accuracy)
    if accuracy > best_score:
        best_score = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f) #save model 

print("Best Score: \n", best_score) """

# load model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: \n", linear.coef_) # Each slope value
print("Intercept: \n", linear.intercept_) # This is the y intercept

predictions = linear.predict(x_test) # Gets a list of all predictions

print("All predictions and actual final grade: \n")
for x in range(len(predictions)):
   print(predictions[x], x_test[x], y_test[x])

#Drawing and plotting model
plot="G1" # can be changed to other attributes such as G1, G2, studytime, failures, absences
style.use("ggplot")
pyplot.scatter(data[plot], data["G3"])
pyplot.legend(loc=4)
pyplot.xlabel(plot)
pyplot.ylabel("Final Grade")
pyplot.show()