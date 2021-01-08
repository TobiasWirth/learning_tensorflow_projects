import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot

from matplotlib import style

data = pd.read_csv("StudentsPerformance.csv", sep=",")

data = data[["gender", "parental level of education", "test preparation course", "math score", "reading score", "writing score"]]


df = pd.DataFrame(data)

df = pd.get_dummies(df, columns=["gender"], prefix=["gender"])

replace_map_education = {"parental level of education": {"high school": 1, "some high school": 1, "associate's degree": 2, "some college": 2,
                                  "bachelor's degree": 3, "master's degree": 4}}

replace_map_preparation = {"test preparation course": {"none": 0, "completed": 1}}

education_labels = df["parental level of education"].astype("category").cat.categories.tolist()
preparation_labels = df["test preparation course"].astype("category").cat.categories.tolist()

replace_map_education_comp = {"parental level of education" : {k: v for k,v in zip(education_labels,list(range(1,len(education_labels)+1)))}}
replace_map_preparation_comp = {"test preparation course" : {k: v for k,v in zip(preparation_labels,list(range(1,len(preparation_labels)+1)))}}

df = df.replace(replace_map_education)
df = df.replace(replace_map_preparation)

"""
for i in range (3):

    if(i == 0):
        predict = "math score"
    elif(i == 1):
        predict = "reading score"
    else:
        predict = "writing score"


    X = np.array(df.drop([predict], 1))  # All Features, drop math score column from table, training data
    y = np.array(df[predict])  # All labels, only math score column

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    best_score = 0

    for _ in range (30):

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)

        accuracy = linear.score(x_test, y_test)

        # save the best model (highest accuracy)
        if accuracy > best_score:
            best_score = accuracy
            if(i == 0):
                with open("math_score_model.pickle", "wb") as f:
                    pickle.dump(linear, f) #save model
            if(i == 1):
                with open("reading_score_model.pickle", "wb") as f:
                    pickle.dump(linear, f) #save model
            else:
                with open("writing_score_model.pickle", "wb") as f:
                    pickle.dump(linear, f)  # save model


    print("Best Score: \n", best_score) """




# load model
pickle_in = open("writing_score_model.pickle", "rb")
linear = pickle.load(pickle_in)

"""
print("Coefficient: \n", linear.coef_)  # Each slope value
print("Intercept: \n", linear.intercept_)  # This is the y intercept

predictions = linear.predict(x_test)  # Gets a list of all predictions

print("All predictions and actual score: \n")
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x]) """

# Drawing and plotting model
plot = "parental level of education"
style.use("ggplot")
pyplot.scatter(data[plot], data["writing score"])
pyplot.legend(loc=4)
pyplot.xlabel(plot)
pyplot.ylabel("Writing Score")
pyplot.show()