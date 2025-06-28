import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dataset = sklearn.datasets.load_iris()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print("DecisionTreeClassifier Results: ", score)