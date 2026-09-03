from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Guess_Who_Dataset.csv",header=None)
dataset = dataset.set_index([0]).transpose()

x_leg = dataset.drop(columns = "Name")
y_leg = dataset["Name"]

x = dataset.drop(columns = "Name")
y = dataset["Name"]

x = pd.get_dummies(x)

label = LabelEncoder()
y = label.fit_transform(y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

print("Data fed to model:")
print(x)

tree.plot_tree(clf,
               feature_names = x.columns.tolist(),
               class_names = label.classes_.tolist(),
               filled = True,
               fontsize = 3
               )
plt.show()


