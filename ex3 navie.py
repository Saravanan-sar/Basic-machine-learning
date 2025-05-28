from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [[1, 20], [2, 21], [3, 22], [4, 23], [5, 24]]
y = [0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GaussianNB()
model.fit(X_train, y_train)
predicted = model.predict(X_test)

print("Predicted:", predicted)
print("Accuracy:", accuracy_score(y_test, predicted))
