from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Guess_Who_Dataset.csv",header=None)
dataset = dataset.set_index([0]).transpose()

print(dataset)

encoder = LabelEncoder()

for col in dataset:
    dataset[col] = encoder.fit_transform(dataset[col])

x = dataset.drop(columns = "Name")
y = dataset["Name"]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

tree.plot_tree(clf)
plt.show()


