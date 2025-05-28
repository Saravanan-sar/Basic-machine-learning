12
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit model to dummy data
import numpy as np
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)
model.fit(X, y, epochs=10, verbose=0)

model.summary()
