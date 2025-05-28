9
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('Countryclusters.csv')
x = data.iloc[:, 1:3]

kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(x)

plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster'], cmap='rainbow')
plt.show()
