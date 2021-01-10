import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print("Features: ", cancer.feature_names)
#print("Labels: ", cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#print(x_train[:5], y_train[:5])

clf_svm = svm.SVC(kernel="linear", C = 2) # select kernel used, C = soft margin level
clf_svm.fit(x_train, y_train)

y_pred = clf_svm.predict(x_test)

svm_accuracy = metrics.accuracy_score(y_test, y_pred)


# compare to knn-classifier

clf_knn = KNeighborsClassifier(n_neighbors=9)

clf_knn.fit(x_train, y_train)

knn_y_pred = clf_knn.predict(x_test)

knn_accuracy = metrics.accuracy_score(y_test, knn_y_pred)

print("SVM Accuracy: ", svm_accuracy, " KNN Accuracy: ", knn_accuracy)


