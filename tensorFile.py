import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

# import data
data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1)) # All Features, drop G3 column from table, training data
y = np.array(data[predict]) # All labels, only G3 column

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

accuracy = linear.score(x_test, y_test)

print("Accuracy: \n", accuracy)
print("Coefficient: \n", linear.coef_) # Each slope value
print("Intercept: \n", linear.intercept_) # This is the y intercept

predictions = linear.predict(x_test) # Gets a list of all predictions

print("All predictions and actual final grade: \n")
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])