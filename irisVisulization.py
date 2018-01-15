#
# Example file from Google Developers: "Visualizing a Decision Tree - Machine Learning Recipes #2": YouTube: https://youtu.be/tNa99PG8hR8
# Category: Supervised Learning
# January 14, 2018
#


# Declarations:
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus


iris = load_iris()
targetLen = len(iris.target)
dataLen = len(iris.data)

# Declarations: Features
featureNames = iris.feature_names

# Declarations: Labels
labelNames = iris.target_names

# Step(1): Collect training data
print("Step(1): Collect training data")
print("Data Lenght: {0}".format(dataLen))
print(iris.data)
print("Target Lenght: {0}".format(targetLen))
print(iris.target)
print("Data Set: ")
for i in range(targetLen):
    print("Data:{0:<25} Label: {1:<25}".format(iris.data[i], labelNames[iris.target[i]]))


# Step(2): Train Classifier: Decision Tree
# Use the decision tree object and then fit 'find' paterns in features and labels
# set random_state=0 to seed the random generator with a consisten number number for testing ONLY
testIndex = [0, 50, 100]
trainData = np.delete(iris.data, testIndex, axis=0)
trainTarget = np.delete(iris.target, testIndex)
print("Test Index: \n{0}".format(testIndex))
print("Training Data: Length: {1}\n{0}".format(trainData, len(trainData)))
print("Training Target: Length: {1}\n{0}".format(trainTarget, len(trainTarget)))
clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainData, trainTarget)


# Step(3): Make Predictions
# the prdict method will return the best fit from the decesion tree
# result = clf.predict([[150, bumpy], [130, smooth], [125.5, bumpy], [110, smooth]])
testData = iris.data[testIndex]
testTarget = iris.target[testIndex]
result = clf.predict(testData)

print("Step(3): Make Predictions: ")
print("Test Data: \n{0}".format(testData))
print("Test Target: {0}".format(testTarget))
print("Test Result: {0}".format(result))
print(featureNames)
for i in range(len(testData)):
    print("Data:{0:<25} Label: {1:<25}".format(testData[i], labelNames[result[i]]))


# Step(4): Visulization
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, impurity=False, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
