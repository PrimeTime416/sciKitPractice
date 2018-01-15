#
# Example file from Google Developers: "Hello World - Machine Learning Recipes": YouTube: https://youtu.be/cKxRvEZd3Mw
# Category: Supervised Learning
# January 14, 2018
#


from sklearn import tree

# Declarations: Texture
bumpy = 0
smooth = 1

# Declarations: Labels
apple = 0
orange = 1


# Step(1): Collect training data
# Features: [Weight, Texture]
features = [[140, smooth], [130, smooth], [150, bumpy], [170, bumpy]]

# labels will be used as the index for the features
labels = [apple, apple, orange, orange]


# Step(2): Train Classifier: Decision Tree
# Use the decision tree object and then fit 'find' paterns in features and labels
# set random_state=0 to seed the random generator with a consisten number number for testing ONLY
clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(features, labels)


# Step(3): Make Predictions
# the prdict method will return the best fit from the decesion tree
result = clf.predict([[150, bumpy], [130, smooth], [125.5, bumpy], [110, smooth]])
# result = clf.predict([[150, bumpy]])
print("Step(3): Make Predictions: ")
for x in result:
    if x == 0:
        print("Apple")
        continue
    elif x == 1:
        print("Orange")
        continue
    print("Orange")
# continue
# print(result)
