from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np

#Link: http://benbrostoff.github.io/2018/01/27/decision-trees-intro.html
# Decision tree learning

#Fake features: minutes, age, height
features = np.array([
    [29, 23, 72],
    [31, 25, 77],
    [31, 27, 82],
    [29, 29, 89],
    [31, 31, 72],
    [29, 33, 77],
])

#We want to label them 0 if they scored under 20 points and 1 if they scored over 20 in a game.
labels = np.array([
    [0], 
    [1], 
    [1], 
    [0],
    [1],
    [0],
])

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.3,
    random_state=42,
)

# Used Gini impurity for  measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled
# Other metrics: Information gain and variance reduction
clf = tree.DecisionTreeClassifier()
clf.fit(X=X_train, y=y_train)

# tree.DecisionTreeClassifier ships with a feature_importances_ that lists the weight of each feature. Because minutes is the only feature that matters, it’s assigned a weight of 1.0, while the meaningless age and height features have a weight of 0.0.
clf.feature_importances_ # [ 1.,  0.,  0.]
clf.score(X=X_test, y=y_test) # 1.0

