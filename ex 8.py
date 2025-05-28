8 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X = np.random.rand(100, 3)
y = np.random.rand(100)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model1 = LinearRegression().fit(X_train, y_train)
model2 = RandomForestRegressor().fit(X_train, y_train)

pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)

final_pred = (pred1 + pred2) / 2
print("MSE:", mean_squared_error(y_test, final_pred))
