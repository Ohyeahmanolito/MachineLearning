from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn import metrics

# Load data
iris = datasets.load_iris()
training_data = iris.data
class_ = iris.target

train_feature, test_feature, train_class, test_class = cross_validation.train_test_split(training_data, class_, test_size=0.33, random_state=5)

'''
# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
'''

# Create one-vs-rest logistic regression object
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')

# Train model
model = clf.fit(train_feature, train_class)

# Predict class
predicted_class = model.predict(test_feature)

cnf_matrix = metrics.confusion_matrix(predicted_class, test_class)
print(cnf_matrix)