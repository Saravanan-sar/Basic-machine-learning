7
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('Social_Network_Ads.csv')
x = data.iloc[:, [0, 1]].values
y = data.iloc[:, 2].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = SVC(kernel='rbf')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
