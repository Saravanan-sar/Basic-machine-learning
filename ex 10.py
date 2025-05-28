10
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

x = np.random.rand(100, 2)

model = GaussianMixture(n_components=2)
model.fit(x)

print("Means:", model.means_)
print("Covariances:", model.covariances_)
