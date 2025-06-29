import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = sklearn.datasets.load_iris()

X = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print("Linear Regression Results: ", score)
