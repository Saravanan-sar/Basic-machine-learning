import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
df['Nationality'] = df['Nationality'].map({'UK': 0, 'USA': 1, 'N': 2})
df['Go'] = df['Go'].map({'YES': 1, 'NO': 0})

features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

plot_tree(clf, feature_names=features)
plt.savefig("tree.png")
